"""
WC2026 Group Stage Predictor
Multi-factor ML-inspired model using:
  1. ELO rating (current strength + form)
  2. Squad average height (set piece threat)
  3. Squad average age × venue temperature (fatigue/heat penalty)
  4. Goalkeeper quality rating
  5. Head-to-head historical record
  6. Underdog motivation factor
  7. Recent form (last 10 matches)
"""

import csv
import math
import random
from collections import defaultdict

random.seed(42)

# ---------------------------------------------------------------------------
# TEAM DATA
# Sources: eloratings.net, transfermarkt, FIFA rankings (as of June 2025)
# ---------------------------------------------------------------------------

TEAMS = {
    # name: {elo, height_cm, avg_age, gk_rating (1-10), recent_form (0-1), confederation}
    "France":               {"elo": 2045, "height": 183.2, "age": 27.1, "gk": 9.2, "form": 0.72, "conf": "UEFA"},
    "Argentina":            {"elo": 2005, "height": 180.1, "age": 27.8, "gk": 9.0, "form": 0.68, "conf": "CONMEBOL"},
    "England":              {"elo": 1981, "height": 183.5, "age": 26.5, "gk": 8.8, "form": 0.70, "conf": "UEFA"},
    "Brazil":               {"elo": 1968, "height": 180.8, "age": 26.2, "gk": 8.5, "form": 0.65, "conf": "CONMEBOL"},
    "Spain":                {"elo": 1962, "height": 181.0, "age": 25.8, "gk": 8.6, "form": 0.72, "conf": "UEFA"},
    "Portugal":             {"elo": 1951, "height": 182.4, "age": 27.9, "gk": 8.4, "form": 0.68, "conf": "UEFA"},
    "Germany":              {"elo": 1940, "height": 183.8, "age": 25.6, "gk": 8.7, "form": 0.65, "conf": "UEFA"},
    "Netherlands":          {"elo": 1905, "height": 185.2, "age": 26.8, "gk": 8.3, "form": 0.67, "conf": "UEFA"},
    "Belgium":              {"elo": 1885, "height": 182.6, "age": 28.4, "gk": 8.1, "form": 0.62, "conf": "UEFA"},
    "Uruguay":              {"elo": 1858, "height": 181.3, "age": 27.0, "gk": 7.8, "form": 0.63, "conf": "CONMEBOL"},
    "Croatia":              {"elo": 1840, "height": 184.1, "age": 29.2, "gk": 8.5, "form": 0.58, "conf": "UEFA"},
    "United States":        {"elo": 1830, "height": 183.0, "age": 25.2, "gk": 7.6, "form": 0.62, "conf": "CONCACAF"},
    "Colombia":             {"elo": 1818, "height": 181.2, "age": 26.4, "gk": 7.8, "form": 0.66, "conf": "CONMEBOL"},
    "Switzerland":          {"elo": 1800, "height": 182.8, "age": 27.6, "gk": 8.2, "form": 0.64, "conf": "UEFA"},
    "Mexico":               {"elo": 1795, "height": 178.4, "age": 26.8, "gk": 7.5, "form": 0.58, "conf": "CONCACAF"},
    "Senegal":              {"elo": 1785, "height": 183.0, "age": 26.5, "gk": 8.0, "form": 0.65, "conf": "CAF"},
    "Japan":                {"elo": 1778, "height": 175.8, "age": 26.2, "gk": 7.9, "form": 0.68, "conf": "AFC"},
    "Morocco":              {"elo": 1770, "height": 181.5, "age": 26.8, "gk": 8.4, "form": 0.64, "conf": "CAF"},
    "Canada":               {"elo": 1752, "height": 183.6, "age": 25.8, "gk": 7.8, "form": 0.60, "conf": "CONCACAF"},
    "South Korea":          {"elo": 1738, "height": 180.2, "age": 27.1, "gk": 7.6, "form": 0.58, "conf": "AFC"},
    "Ecuador":              {"elo": 1725, "height": 179.8, "age": 25.5, "gk": 7.4, "form": 0.56, "conf": "CONMEBOL"},
    "Ivory Coast":          {"elo": 1712, "height": 181.8, "age": 27.4, "gk": 7.5, "form": 0.58, "conf": "CAF"},
    "Australia":            {"elo": 1705, "height": 182.4, "age": 27.0, "gk": 7.6, "form": 0.56, "conf": "AFC"},
    "Sweden":               {"elo": 1698, "height": 185.0, "age": 27.8, "gk": 7.8, "form": 0.60, "conf": "UEFA"},
    "Norway":               {"elo": 1690, "height": 184.2, "age": 26.0, "gk": 7.4, "form": 0.58, "conf": "UEFA"},
    "Algeria":              {"elo": 1675, "height": 181.4, "age": 27.2, "gk": 7.5, "form": 0.55, "conf": "CAF"},
    "Austria":              {"elo": 1665, "height": 183.0, "age": 26.5, "gk": 7.6, "form": 0.60, "conf": "UEFA"},
    "Turkiye":              {"elo": 1658, "height": 183.2, "age": 26.8, "gk": 7.7, "form": 0.58, "conf": "UEFA"},
    "Paraguay":             {"elo": 1645, "height": 179.5, "age": 26.2, "gk": 7.2, "form": 0.52, "conf": "CONMEBOL"},
    "Iran":                 {"elo": 1638, "height": 183.8, "age": 28.2, "gk": 7.8, "form": 0.56, "conf": "AFC"},
    "Tunisia":              {"elo": 1625, "height": 181.6, "age": 27.5, "gk": 7.3, "form": 0.54, "conf": "CAF"},
    "Egypt":                {"elo": 1615, "height": 181.2, "age": 27.8, "gk": 7.4, "form": 0.55, "conf": "CAF"},
    "Ghana":                {"elo": 1602, "height": 181.0, "age": 26.0, "gk": 7.0, "form": 0.50, "conf": "CAF"},
    "Bosnia and Herzegovina": {"elo": 1595, "height": 184.2, "age": 27.5, "gk": 7.2, "form": 0.52, "conf": "UEFA"},
    "Scotland":             {"elo": 1580, "height": 183.5, "age": 27.8, "gk": 7.4, "form": 0.54, "conf": "UEFA"},
    "Saudi Arabia":         {"elo": 1568, "height": 180.8, "age": 26.5, "gk": 7.0, "form": 0.50, "conf": "AFC"},
    "Panama":               {"elo": 1548, "height": 181.2, "age": 27.2, "gk": 7.0, "form": 0.50, "conf": "CONCACAF"},
    "Jordan":               {"elo": 1532, "height": 180.5, "age": 26.8, "gk": 6.8, "form": 0.48, "conf": "AFC"},
    "New Zealand":          {"elo": 1515, "height": 183.8, "age": 26.5, "gk": 6.8, "form": 0.46, "conf": "OFC"},
    "DR Congo":             {"elo": 1505, "height": 182.2, "age": 26.2, "gk": 6.9, "form": 0.48, "conf": "CAF"},
    "Uzbekistan":           {"elo": 1492, "height": 181.8, "age": 25.5, "gk": 6.7, "form": 0.50, "conf": "AFC"},
    "Cape Verde":           {"elo": 1478, "height": 181.0, "age": 26.8, "gk": 6.8, "form": 0.52, "conf": "CAF"},
    "Czechia":              {"elo": 1748, "height": 184.0, "age": 27.8, "gk": 7.9, "form": 0.58, "conf": "UEFA"},
    "Iraq":                 {"elo": 1420, "height": 180.2, "age": 26.5, "gk": 6.6, "form": 0.46, "conf": "AFC"},
    "South Africa":         {"elo": 1412, "height": 181.5, "age": 26.8, "gk": 6.8, "form": 0.48, "conf": "CAF"},
    "Curacao":              {"elo": 1395, "height": 181.0, "age": 27.5, "gk": 6.5, "form": 0.44, "conf": "CONCACAF"},
    "Qatar":                {"elo": 1380, "height": 179.0, "age": 26.2, "gk": 6.4, "form": 0.40, "conf": "AFC"},
    "Haiti":                {"elo": 1355, "height": 180.5, "age": 26.5, "gk": 6.2, "form": 0.38, "conf": "CONCACAF"},
}

# ---------------------------------------------------------------------------
# HEAD-TO-HEAD ADVANTAGE MAP
# Positive = team_a has historical edge over team_b
# Based on notable historical rivalries and patterns
# ---------------------------------------------------------------------------

H2H_ADVANTAGE = {
    # (team_a, team_b): advantage for team_a (-1 to +1)
    ("Argentina", "Brazil"):        0.05,
    ("Germany", "Netherlands"):     0.10,
    ("Spain", "Portugal"):          0.08,
    ("England", "Germany"):        -0.05,
    ("Uruguay", "Argentina"):      -0.05,
    ("France", "England"):          0.08,
    ("Mexico", "United States"):    0.12,
    ("Germany", "England"):         0.10,
    ("Brazil", "Argentina"):       -0.05,
    ("Morocco", "Algeria"):         0.05,
    ("South Korea", "Japan"):       0.03,
    ("Iran", "Saudi Arabia"):       0.05,
    ("Egypt", "Algeria"):           0.04,
    ("Netherlands", "Germany"):    -0.10,
    ("Croatia", "Bosnia and Herzegovina"): 0.15,
    ("Spain", "Netherlands"):       0.10,
}


def get_h2h(team1, team2):
    """Return H2H advantage for team1 over team2."""
    if (team1, team2) in H2H_ADVANTAGE:
        return H2H_ADVANTAGE[(team1, team2)]
    if (team2, team1) in H2H_ADVANTAGE:
        return -H2H_ADVANTAGE[(team2, team1)]
    return 0.0


# ---------------------------------------------------------------------------
# VENUE TEMPERATURES (average °C during June-July 2026)
# Groups will play across multiple venues; we assign approximate temps
# ---------------------------------------------------------------------------

VENUE_TEMPS = {
    # city: avg_temp_celsius
    "Dallas":       33,
    "Houston":      35,
    "Miami":        31,
    "Atlanta":      29,
    "Kansas City":  27,
    "New York":     24,
    "Los Angeles":  22,
    "San Francisco":18,
    "Seattle":      20,
    "Boston":       22,
    "Guadalajara":  27,
    "Mexico City":  20,
    "Monterrey":    34,
    "Toronto":      23,
    "Vancouver":    20,
}

# Group → typical venue temperature (weighted mix of assigned venues)
GROUP_TEMPS = {
    "A": 28,   # Mexico City + Dallas + San Francisco
    "B": 22,   # Toronto + Kansas City + Seattle
    "C": 27,   # Los Angeles + Guadalajara + Houston
    "D": 25,   # Kansas City + New York + Dallas
    "E": 24,   # Philadelphia + Atlanta + New York
    "F": 22,   # Seattle + Los Angeles + San Francisco
    "G": 28,   # Dallas + Atlanta + Miami
    "H": 25,   # Boston + New York + Los Angeles
    "I": 26,   # Houston + Monterrey + Miami
    "J": 24,   # Los Angeles + Seattle + Vancouver
    "K": 28,   # Guadalajara + Mexico City + Monterrey
    "L": 23,   # New York + Boston + Atlanta
}

# Average home temperature per confederation (used for heat penalty)
CONF_HOME_TEMP = {
    "UEFA":     12,   # European average
    "CONMEBOL": 22,   # South American average (but varies)
    "CAF":      28,   # African average
    "AFC":      25,   # Asian average
    "CONCACAF": 25,
    "OFC":      15,
}

# Override specific teams with known extreme home climates
TEAM_HOME_TEMP = {
    "Saudi Arabia": 38,
    "Qatar":        38,
    "Iraq":         40,
    "Iran":         28,
    "Egypt":        30,
    "Morocco":      22,
    "Senegal":      30,
    "Ivory Coast":  30,
    "Ghana":        29,
    "Algeria":      25,
    "Tunisia":      22,
    "Cape Verde":   25,
    "Haiti":        30,
    "Curacao":      30,
    "Mexico":       22,
    "Panama":       28,
    "Canada":       10,
    "Sweden":        8,
    "Norway":        6,
    "New Zealand":  14,
    "Scotland":     10,
    "Australia":    18,
    "Japan":        20,
    "South Korea":  18,
    "Ecuador":      18,
    "Colombia":     22,
    "Uruguay":      15,
    "Paraguay":     25,
    "Brazil":       26,
    "Argentina":    15,
    "United States": 20,
    "Czechia":      12,
    "Croatia":      14,
    "Bosnia and Herzegovina": 13,
    "Switzerland":  10,
    "Austria":      10,
    "Turkey":       18,
    "Turkiye":      18,
    "Jordan":       28,
    "Uzbekistan":   25,
    "DR Congo":     28,
    "South Africa": 18,
}


def get_home_temp(team):
    if team in TEAM_HOME_TEMP:
        return TEAM_HOME_TEMP[team]
    conf = TEAMS[team]["conf"]
    return CONF_HOME_TEMP.get(conf, 20)


def heat_penalty(team, group_temp):
    """
    Older squads in unfamiliar heat perform worse.
    Penalty = 0.01 per degree above home temp, scaled by age factor.
    Age factor: linearly increases above 26 years old.
    """
    home_temp = get_home_temp(team)
    temp_diff = group_temp - home_temp
    if temp_diff <= 0:
        return 0.0  # Playing in cooler-than-home conditions → no penalty

    avg_age = TEAMS[team]["age"]
    age_factor = max(0, (avg_age - 25.5) / 4.0)  # 0 at 25.5, 1 at 29.5

    # penalty on expected goals: up to ~0.15 for very old squad in extreme heat
    penalty = temp_diff * 0.004 * (1 + age_factor)
    return min(penalty, 0.20)


# ---------------------------------------------------------------------------
# STRENGTH CALCULATION
# Combines ELO, height (set pieces), form, GK quality
# ---------------------------------------------------------------------------

ELO_BASE = 1500


def elo_win_prob(elo_a, elo_b):
    """Standard ELO win probability."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def height_bonus(team):
    """Taller teams score ~0.05 more from set pieces per 5cm above avg."""
    avg_height = 181.5
    bonus = (TEAMS[team]["height"] - avg_height) / 5.0 * 0.05
    return max(-0.1, min(0.1, bonus))


def underdog_motivation(elo_a, elo_b):
    """
    When team A is a big underdog (low win prob), they get a motivation boost.
    Based on idea that underdogs are more fired up.
    Max boost of ~0.1 expected goals when win prob < 20%.
    """
    prob = elo_win_prob(elo_a, elo_b)
    if prob < 0.2:
        return 0.08 * (0.2 - prob) / 0.2
    return 0.0


def gk_defensive_factor(team):
    """GK quality affects goals conceded. 7.5 = average (factor 1.0)."""
    gk = TEAMS[team]["gk"]
    return 1.0 - (gk - 7.5) * 0.04  # 9.5 GK → 0.92 factor; 6.0 GK → 1.06 factor


# ---------------------------------------------------------------------------
# EXPECTED GOALS MODEL
# ---------------------------------------------------------------------------

LEAGUE_AVG_GOALS = 1.35  # average goals per team per match in international football


def expected_goals(team_a, team_b, group):
    """
    Compute expected goals for team_a when playing team_b in given group.
    """
    t_a = TEAMS[team_a]
    t_b = TEAMS[team_b]

    group_temp = GROUP_TEMPS.get(group, 25)

    # Base: ELO-derived attack/defense balance
    elo_ratio = t_a["elo"] / t_b["elo"]
    base_xg = LEAGUE_AVG_GOALS * (elo_ratio ** 1.5)

    # Form adjustment
    form_adj = (t_a["form"] - 0.55) * 0.3  # ±0.15
    base_xg += form_adj

    # Height / set piece bonus
    base_xg += height_bonus(team_a)

    # Head-to-head
    h2h = get_h2h(team_a, team_b)
    base_xg += h2h * 0.2

    # Underdog motivation (for team_a if underdog)
    base_xg += underdog_motivation(t_a["elo"], t_b["elo"])

    # Heat penalty on attack (tired legs = fewer goals)
    base_xg -= heat_penalty(team_a, group_temp)

    # Opponent GK reduces scoring
    opp_gk_factor = gk_defensive_factor(team_b)
    base_xg *= opp_gk_factor

    return max(0.3, base_xg)


# ---------------------------------------------------------------------------
# SCORE SIMULATION via Poisson
# ---------------------------------------------------------------------------

def poisson_pmf(k, lam):
    """P(X=k) for Poisson(lam)."""
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def predict_score(team1, team2, group):
    """
    Predict match score using expected goals.
    Uses rounded expected value (mean of Poisson = lambda) rather than
    the distribution mode, which would collapse all evenly-matched games to 1-1.
    Adds a small ELO-based tie-breaking adjustment.
    """
    xg1 = expected_goals(team1, team2, group)
    xg2 = expected_goals(team2, team1, group)

    # Determine result type via win probabilities
    elo1 = TEAMS[team1]["elo"]
    elo2 = TEAMS[team2]["elo"]
    p_win1 = elo_win_prob(elo1, elo2)

    # Base scores from rounded xg (Poisson mean = lambda)
    s1 = int(round(xg1))
    s2 = int(round(xg2))

    # Tie-break: if rounded scores are equal but one team is clearly stronger,
    # push the stronger team up by 1 (avoids implausible all-draw groups)
    if s1 == s2:
        if p_win1 > 0.62:       # strong favourite for team1
            s1 += 1
        elif p_win1 < 0.38:     # strong favourite for team2
            s2 += 1
        # between 0.38–0.62 → genuine draw is fine

    return s1, s2


# ---------------------------------------------------------------------------
# GROUP STAGE SIMULATION
# ---------------------------------------------------------------------------

MATCHES = [
    # match_id, group, team1, team2
    (1,  "A", "Mexico",               "South Africa"),
    (2,  "A", "Mexico",               "South Korea"),
    (3,  "A", "Mexico",               "Czechia"),
    (4,  "A", "South Africa",         "South Korea"),
    (5,  "A", "South Africa",         "Czechia"),
    (6,  "A", "South Korea",          "Czechia"),
    (7,  "B", "Canada",               "Bosnia and Herzegovina"),
    (8,  "B", "Canada",               "Qatar"),
    (9,  "B", "Canada",               "Switzerland"),
    (10, "B", "Bosnia and Herzegovina", "Qatar"),
    (11, "B", "Bosnia and Herzegovina", "Switzerland"),
    (12, "B", "Qatar",                "Switzerland"),
    (13, "C", "Brazil",               "Morocco"),
    (14, "C", "Brazil",               "Haiti"),
    (15, "C", "Brazil",               "Scotland"),
    (16, "C", "Morocco",              "Haiti"),
    (17, "C", "Morocco",              "Scotland"),
    (18, "C", "Haiti",                "Scotland"),
    (19, "D", "United States",        "Paraguay"),
    (20, "D", "United States",        "Australia"),
    (21, "D", "United States",        "Turkiye"),
    (22, "D", "Paraguay",             "Australia"),
    (23, "D", "Paraguay",             "Turkiye"),
    (24, "D", "Australia",            "Turkiye"),
    (25, "E", "Germany",              "Curacao"),
    (26, "E", "Germany",              "Ivory Coast"),
    (27, "E", "Germany",              "Ecuador"),
    (28, "E", "Curacao",              "Ivory Coast"),
    (29, "E", "Curacao",              "Ecuador"),
    (30, "E", "Ivory Coast",          "Ecuador"),
    (31, "F", "Netherlands",          "Japan"),
    (32, "F", "Netherlands",          "Sweden"),
    (33, "F", "Netherlands",          "Tunisia"),
    (34, "F", "Japan",                "Sweden"),
    (35, "F", "Japan",                "Tunisia"),
    (36, "F", "Sweden",               "Tunisia"),
    (37, "G", "Belgium",              "Egypt"),
    (38, "G", "Belgium",              "Iran"),
    (39, "G", "Belgium",              "New Zealand"),
    (40, "G", "Egypt",                "Iran"),
    (41, "G", "Egypt",                "New Zealand"),
    (42, "G", "Iran",                 "New Zealand"),
    (43, "H", "Spain",                "Cape Verde"),
    (44, "H", "Spain",                "Saudi Arabia"),
    (45, "H", "Spain",                "Uruguay"),
    (46, "H", "Cape Verde",           "Saudi Arabia"),
    (47, "H", "Cape Verde",           "Uruguay"),
    (48, "H", "Saudi Arabia",         "Uruguay"),
    (49, "I", "France",               "Senegal"),
    (50, "I", "France",               "Iraq"),
    (51, "I", "France",               "Norway"),
    (52, "I", "Senegal",              "Iraq"),
    (53, "I", "Senegal",              "Norway"),
    (54, "I", "Iraq",                 "Norway"),
    (55, "J", "Argentina",            "Algeria"),
    (56, "J", "Argentina",            "Austria"),
    (57, "J", "Argentina",            "Jordan"),
    (58, "J", "Algeria",              "Austria"),
    (59, "J", "Algeria",              "Jordan"),
    (60, "J", "Austria",              "Jordan"),
    (61, "K", "Portugal",             "DR Congo"),
    (62, "K", "Portugal",             "Uzbekistan"),
    (63, "K", "Portugal",             "Colombia"),
    (64, "K", "DR Congo",             "Uzbekistan"),
    (65, "K", "DR Congo",             "Colombia"),
    (66, "K", "Uzbekistan",           "Colombia"),
    (67, "L", "England",              "Croatia"),
    (68, "L", "England",              "Ghana"),
    (69, "L", "England",              "Panama"),
    (70, "L", "Croatia",              "Ghana"),
    (71, "L", "Croatia",              "Panama"),
    (72, "L", "Ghana",                "Panama"),
]


def simulate_group(group_matches, group_id):
    """Simulate all matches in a group, return standings."""
    results = []
    for match_id, grp, team1, team2 in group_matches:
        s1, s2 = predict_score(team1, team2, grp)
        results.append((match_id, grp, team1, team2, s1, s2))
    return results


def compute_standings(group_results):
    """Return sorted standings for a group."""
    teams = {}
    for _, _, t1, t2, s1, s2 in group_results:
        for t in [t1, t2]:
            if t not in teams:
                teams[t] = {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0}

        if s1 > s2:
            teams[t1]["pts"] += 3; teams[t1]["w"] += 1
            teams[t2]["l"] += 1
        elif s1 < s2:
            teams[t2]["pts"] += 3; teams[t2]["w"] += 1
            teams[t1]["l"] += 1
        else:
            teams[t1]["pts"] += 1; teams[t1]["d"] += 1
            teams[t2]["pts"] += 1; teams[t2]["d"] += 1

        teams[t1]["gf"] += s1; teams[t1]["ga"] += s2; teams[t1]["gd"] += s1 - s2
        teams[t2]["gf"] += s2; teams[t2]["ga"] += s1; teams[t2]["gd"] += s2 - s1

    sorted_teams = sorted(teams.items(), key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]))
    return sorted_teams


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("WC2026 GROUP STAGE PREDICTIONS")
    print("=" * 60)

    all_results = []
    group_map = defaultdict(list)

    for match in MATCHES:
        group_map[match[1]].append(match)

    for group_id in sorted(group_map.keys()):
        results = simulate_group(group_map[group_id], group_id)
        all_results.extend(results)

        print(f"\n--- Group {group_id} ---")
        for _, _, t1, t2, s1, s2 in results:
            outcome = "WIN" if s1 > s2 else ("DRAW" if s1 == s2 else "LOSS")
            print(f"  {t1:25s} {s1} - {s2}  {t2:25s}  [{outcome} for {t1}]")

        standings = compute_standings(results)
        print(f"\n  Standings:")
        for i, (team, st) in enumerate(standings):
            qualifier = "✓ QUALIFIES" if i < 2 else "  (3rd - may qualify)"
            print(f"  {i+1}. {team:25s}  {st['pts']}pts  GD:{st['gd']:+d}  GF:{st['gf']}  {qualifier}")

    # Write output CSV
    output_path = "/Users/melswittenberg/Documents/github/WC26 - prediction model/output.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["match_id", "group", "team1", "team2", "score1", "score2"])
        for match_id, grp, t1, t2, s1, s2 in all_results:
            writer.writerow([match_id, grp, t1, t2, s1, s2])

    print(f"\n\nOutput written to: {output_path}")
    print("\n--- Predicted Group Qualifiers (Top 2 per group) ---")
    for group_id in sorted(group_map.keys()):
        group_results = [r for r in all_results if r[1] == group_id]
        standings = compute_standings(group_results)
        qualifiers = [standings[0][0], standings[1][0]]
        thirds = standings[2][0] if len(standings) > 2 else "?"
        print(f"  Group {group_id}: {qualifiers[0]} & {qualifiers[1]}  (3rd: {thirds})")


if __name__ == "__main__":
    main()
