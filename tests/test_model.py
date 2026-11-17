import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fpl.model import FPLModel, FeatureHistory, _evaluate, _observations, forecast


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


if __name__ == "__main__":
    unittest.main()
