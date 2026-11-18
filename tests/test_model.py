import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fpl.model import FPLModel, FeatureHistory, _evaluate, _observations, _predict_fixtures, forecast, forecast_gameweeks


def match(fixture, gameweek, kickoff, home=1, away=2, points=2, finished=True, season=2026):
    date = pd.Timestamp(kickoff, tz="UTC")
    return {
        "season": season, "gameweek": gameweek, "fixture": fixture, "kickoff": date,
        "deadline": date.replace(hour=12, minute=0, second=0), "home_team": home, "away_team": away,
        "home_goals": points, "away_goals": 1, "finished": finished,
        "home_difficulty": 3, "away_difficulty": 4,
    }


def outcome(match, points, minutes=90):
    return {
        **match, "player": 12345, "position": "MID", "team": match["home_team"],
        "opponent": match["away_team"], "home": True, "points": points, "minutes": minutes,
        "goals_scored": points, "bps": points * 2,
    }


class AlwaysStarts:
    classes_ = np.array([4])

    def predict_proba(self, features):
        return np.ones((len(features), 1))


class VenuePoints:
    def predict(self, features):
        return 4 + 3 * features["home"].to_numpy()


class LearnsFlags:
    classes_ = np.array([0, 4])

    def predict_proba(self, features):
        available = features["flag_probability"].fillna(1).to_numpy()
        return np.column_stack((1 - available, available))


class RecentPoints:
    def predict(self, features):
        return features["points_3"].fillna(2).to_numpy()


def bootstrap():
    original = {"id": 99, "web_name": "Example", "code": 12345, "team": 1, "element_type": 3, "status": "a"}
    blank = {**original, "id": 100, "code": 789, "team": 4, "element_type": 1}
    injured = {**original, "id": 101, "code": 345, "chance_of_playing_next_round": 50,
               "chance_of_playing_this_round": 100, "status": "d"}
    return {
        "events": [{"id": 3, "is_next": True}],
        "elements": [original, blank, injured],
        "teams": [{"id": club, "short_name": f"T{club}", "name": f"Team {club}"} for club in range(1, 5)],
    }


class FeatureHistoryTests(unittest.TestCase):
    def test_future_and_target_outcomes_and_late_previous_matches_do_not_leak(self):
        earlier = match(1, 1, "2026-08-08 15:00", home=2, away=1, points=2)
        late_previous = match(2, 2, "2026-08-22 11:00", home=2, away=1, points=99)
        target = match(3, 3, "2026-08-22 15:00")
        # A target Gameweek game with an old kickoff still cannot enter the snapshot.
        early_target = match(4, 3, "2026-08-21 19:00", home=2, away=1, points=99)
        later = match(5, 4, "2026-08-29 15:00", home=2, away=1, points=99)
        unfinished = match(6, 2, "2026-08-20 15:00", home=2, away=1, points=99, finished=False)
        history = [outcome(earlier, 5), *(outcome(fixture, 99) for fixture in (late_previous, target, early_target, later, unfinished))]
        targets = pd.DataFrame([outcome(target, 0)])
        baseline = FeatureHistory(pd.DataFrame(history[:1]), pd.DataFrame([earlier, target]))
        full = FeatureHistory(pd.DataFrame(history), pd.DataFrame([earlier, late_previous, target, early_target, later, unfinished]))
        expected = baseline.snapshot(targets, 2026, 3, target["deadline"])
        actual = full.snapshot(targets, 2026, 3, target["deadline"])
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(actual.iloc[0]["points_3"], 5)
        self.assertEqual(actual.iloc[0]["opponent_goals_for_20"], 2)
        self.assertTrue(pd.isna(actual.iloc[0]["expected_goals_20"]))

    def test_new_context_and_team_xg_are_deadline_safe(self):
        prior = match(1, 1, "2026-08-08 15:00")
        target = match(2, 2, "2026-08-15 15:00")
        cutoff = target["deadline"]
        home = dict(outcome(prior, 3), expected_goals=0.4)
        away = dict(outcome(prior, 4), home=False, team=2, opponent=1, player=678, expected_goals=1.3)
        result = dict(outcome(target, 10), expected_goals=99)
        roster = pd.DataFrame([
            {"season": 2026, "gameweek": 2, "player": 12345, "context_asof": timestamp,
             "flag_probability": chance, "penalty_order": rank}
            for timestamp, chance, rank in ((cutoff - pd.Timedelta(hours=1), 0.75, 2), (cutoff, 0, 1))
        ])
        quotes = pd.DataFrame([
            {"season": 2026, "fixture": 2, "market_asof": timestamp,
             "home_expected_goals": rate, "away_expected_goals": 2.4, "home_clean_sheet_probability": 0.25}
            for timestamp, rate in ((cutoff - pd.Timedelta(hours=2), 1.4), (cutoff, 4.2))
        ])
        fixtures = pd.DataFrame([prior, target]).assign(home_team_code=3, away_team_code=7)
        workload = pd.DataFrame({"team_code": [3, 3], "kickoff": [
            cutoff - pd.Timedelta(days=3), cutoff - pd.Timedelta(hours=1)
        ]})
        workload.attrs.update(covered_years=(2026,), club_codes=(3, 7))
        builder = FeatureHistory(
            pd.DataFrame([home, away, result]), fixtures, player_context=roster,
            market_context=quotes, other_fixtures=workload,
        )
        row = builder.snapshot(pd.DataFrame([result]), 2026, 2, cutoff).iloc[0]
        self.assertEqual(row["team_xg_for_6"], 0.4)
        self.assertEqual(row["team_xg_against_6"], 1.3)
        self.assertEqual(row["opponent_xg_for_20"], 1.3)
        self.assertEqual(row["flag_probability"], 0.75)
        self.assertEqual(row["penalty_order"], 2)
        self.assertEqual(row["market_team_xg"], 1.4)
        self.assertEqual(row["market_opponent_xg"], 2.4)
        self.assertEqual(row["other_matches_7"], 1)

    def test_post_deadline_transfer_does_not_supply_old_club_features(self):
        fixture = match(1, 1, "2026-08-08 15:00")
        context = pd.DataFrame([{
            "season": 2026, "gameweek": 1, "player": 12345, "context_team": 2,
            "context_position": "MID", "context_asof": fixture["deadline"] - pd.Timedelta(hours=1),
        }])
        rows = pd.DataFrame([outcome(fixture, 5), dict(outcome(fixture, 4), player=99)])
        builder = FeatureHistory(rows, pd.DataFrame([fixture]), player_context=context)
        self.assertEqual(_observations(builder)["_player"].tolist(), [99])

    def test_training_cutoff_excludes_unfinished_and_not_yet_known_labels(self):
        complete = match(1, 1, "2026-08-08 15:00")
        unfinished = match(2, 2, "2026-08-15 15:00", finished=False)
        future = match(3, 3, "2026-08-22 15:00")
        scheduled = [complete, unfinished, future]
        builder = FeatureHistory(pd.DataFrame([outcome(fixture, 2) for fixture in scheduled]), pd.DataFrame(scheduled))
        labels = _observations(builder, as_of=future["deadline"])
        self.assertEqual(labels["_gameweek"].tolist(), [1])

    def test_validation_models_only_fit_on_earlier_seasons_and_aggregate_doubles(self):
        matches = [match(1, 1, f"{year}-08-08 15:00", season=year) for year in (2023, 2024, 2025)]
        matches.append(match(2, 1, "2025-08-11 15:00", season=2025))
        matches[-1]["deadline"] = matches[-2]["deadline"]
        builder = FeatureHistory(pd.DataFrame([outcome(fixture, 1) for fixture in matches]), pd.DataFrame(matches))
        rows = _observations(builder)
        trained_on = []

        def record_train(training_data, features):
            trained_on.append(sorted(training_data["_season"].unique().tolist()))
            return FPLModel(AlwaysStarts(), VenuePoints(), ("home",), {}, len(training_data), trained_on[-1])

        with patch("fpl.model._fit", side_effect=record_train):
            result = _evaluate(rows, ("home",))
        self.assertEqual(trained_on, [[2023], [2023, 2024]])
        self.assertEqual([fold["season"] for fold in result["folds"]], [2024, 2025])
        self.assertEqual([fold["rows"] for fold in result["folds"]], [1, 1])
        self.assertAlmostEqual(result["folds"][-1]["mae"], 12)


class ForecastTests(unittest.TestCase):
    def test_minutes_bands_blend_conditional_points_and_expected_minutes(self):
        class MixedMinutes:
            classes_ = np.arange(5)

            def predict_proba(self, inputs):
                return np.array([[0.1, 0.3, 0.2, 0.1, 0.3]])

        class BandPoints:
            def predict(self, inputs):
                return 1 + 5 * inputs["played_60"].to_numpy()

        fitted = FPLModel(MixedMinutes(), BandPoints(), ("home",), {}, 1, [2026],
                          minutes_by_outcome={0: 0, 1: 20, 2: 40, 3: 70, 4: 90}, points_by_minutes_band=True)
        scored = _predict_fixtures(fitted, pd.DataFrame({"home": [1]})).iloc[0]
        self.assertAlmostEqual(scored["points"], 2.9)
        self.assertAlmostEqual(scored["minutes"], 48)
        self.assertAlmostEqual(scored["appearance"], 0.9)
        self.assertAlmostEqual(scored["sixty"], 0.4)

    def test_transfer_horizon_uses_one_information_cutoff_and_round_specific_availability(self):
        prior = match(1, 1, "2026-08-08 15:00")
        upcoming = match(2, 3, "2026-08-22 15:00", finished=False)
        later_result = match(3, 3, "2026-08-23 15:00")
        future = [
            match(week, week, kickoff, finished=False)
            for week, kickoff in enumerate((
                "2026-08-29 15:00", "2026-09-05 15:00", "2026-09-12 15:00",
                "2026-09-19 15:00", "2026-09-26 15:00",
            ), start=4)
        ]
        calendar = bootstrap()
        calendar["events"] += [
            {"id": fixture["gameweek"], "deadline_time": fixture["deadline"].isoformat()}
            for fixture in future
        ]
        fitted = FPLModel(AlwaysStarts(), RecentPoints(), ("points_3",), {}, 1, [2026])
        weeks = forecast_gameweeks(
            fitted, pd.DataFrame([outcome(prior, 5), outcome(later_result, 99)]),
            pd.DataFrame([prior, upcoming, later_result, *future]), calendar, 2026, 3, upcoming["deadline"],
        )
        self.assertEqual(list(weeks), [3, 4, 5, 6, 7])
        self.assertEqual([week[99]["points"] for week in weeks.values()], [5] * 5)
        self.assertEqual([week[101]["points"] for week in weeks.values()], [1, 2, 2, 2, 2])
        self.assertEqual([week[100]["points"] for week in weeks.values()], [0] * 5)

    def test_double_and_blank_fixtures_and_official_availability(self):
        prior = match(1, 1, "2026-08-08 15:00")
        first = match(2, 3, "2026-08-22 15:00", finished=False)
        second = match(3, 3, "2026-08-25 16:00", home=3, away=1, finished=False)
        finished_target_fixture = match(4, 3, "2026-08-28 17:00", finished=True)
        model = FPLModel(AlwaysStarts(), VenuePoints(), ("home",), {}, 1, [2026])
        predictions = forecast(
            model, pd.DataFrame([outcome(prior, 5)]),
            pd.DataFrame([prior, first, second, finished_target_fixture]), bootstrap(), 2026, 3, first["deadline"],
        )
        self.assertEqual(predictions[99]["points"], 11.0)
        self.assertEqual([fixture["points"] for fixture in predictions[99]["fixtures"]], [7, 4])
        self.assertEqual([fixture["opponent"] for fixture in predictions[99]["fixtures"]], ["T2", "T3"])
        self.assertEqual(predictions[100]["points"], 0)
        self.assertEqual(predictions[100]["fixtures"], [])
        self.assertEqual(predictions[100]["appearance_chance"], 0)
        self.assertEqual(predictions[101]["points"], 5.5)
        self.assertEqual(predictions[101]["start_chance"], 0.5)
        self.assertEqual(predictions[101]["sixty_chance"], 0.5)

    def test_model_using_archived_flags_does_not_apply_live_flag_twice(self):
        prior = match(1, 1, "2026-08-08 15:00")
        fixture = match(2, 3, "2026-08-22 15:00", finished=False)
        cutoff = fixture["deadline"]
        context = pd.DataFrame([{
            "season": 2026, "gameweek": 3, "player": 345,
            "context_asof": cutoff - pd.Timedelta(hours=1), "flag_probability": 0.5,
        }])
        forecast_model = FPLModel(LearnsFlags(), VenuePoints(), ("home", "flag_probability"), {}, 1, [2026])
        predictions = forecast(
            forecast_model, pd.DataFrame([outcome(prior, 5)]), pd.DataFrame([prior, fixture]),
            bootstrap(), 2026, 3, cutoff, player_context=context,
        )
        self.assertEqual(predictions[101]["points"], 3.5)
        self.assertEqual(predictions[101]["appearance_chance"], 0.5)


if __name__ == "__main__":
    unittest.main()
