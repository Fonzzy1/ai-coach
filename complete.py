#! python3
from openai import OpenAI
import itertools
import json
from copy import deepcopy
import numpy as np


client = OpenAI()

gpt_assistant_prompt = """
You are a scheduling robot that can help athletes work out how to schedule their week
You will be sent a Json object with the following format:
{
  fixed_events: {
    MON: {AM: None, PM: None},
    TUE: {AM: None, PM: None},
    WED: {AM: None, PM: None},
    THU: {AM: None, PM: None},
    FRI: {AM: {type: Type1, name: name 2, intensity: 3}, PM: None},
    SAT: {AM: None, PM: None},
    SUN: {AM: None, PM: None}
  },
  additional_events: {
    Type1: [
      {name: name 1, intensity: 1}
    ],
    Type2: []
  },
  schedule: {
      day type 1: {count: , limit: }
  }
  rules: []
}

All events will have a name, a type and intensity. Your job is to take all the additional events and insert them into the weekly schedule in a matter that will match the rules given to you in the 'rules' section.
Intestity is a value between 1 and 5, with 1 being the lowest and 5 being the highest.
Two events can have the same name, type and intensity, make sure to include them both.

The schedule section gives types of days
The count is them number of days per week which need to be allocated this type
The limit is the maximum total intensity for events scheduled on that day type

Give your response in the form of a json object with the following format without any additional text or explanation:
{
    MON: {AM: None, PM: None, DAY_TYPE: None},
    TUE: {AM: None, PM: None, DAY_TYPE: None},
    WED: {AM: None, PM: None, DAY_TYPE: None},
    THU: {AM: None, PM: None, DAY_TYPE: None},
    FRI: {AM: {type: Type1, name: name 2, intensity: 3}, PM: None, DAY_TYPE: None},
    SAT: {AM: None, PM: None, DAY_TYPE},
    SUN: {AM: None, PM: None, DAY_TYPE: None}
}

"""


def make_initial(reqs: Requirements):
    message = [
        {"role": "system", "content": gpt_assistant_prompt},
        {"role": "user", "content": json.dumps(reqs.prompt_dict)},
    ]
    temperature = 0.01
    frequency_penalty = 0.0

    response = client.chat.completions.create(
        model="gpt-4",
        messages=message,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
    )
    return json.loads(response.choices[0].message.content)


class Requirements:
    def __init__(self):
        self.prompt_dict = {
            "fixed_events": {
                "MON": {"AM": None, "PM": None},
                "TUE": {"AM": None, "PM": None},
                "WED": {"AM": None, "PM": None},
                "THU": {"AM": None, "PM": None},
                "FRI": {"AM": None, "PM": None},
                "SAT": {"AM": None, "PM": None},
                "SUN": {"AM": None, "PM": None},
            },
            "schedule": {"rest": {"count": 7, "limit": 0}},
            "additional_events": {},
            "rules": [],
        }
        self.types = []
        self.events = 0
        self.weekly_threshold = 0

    def add_fixed_event(self, day, time, type, name, intensity):
        if type not in self.types:
            self.add_type(type)
        self.prompt_dict["fixed_events"][day][time] = {
            "type": type,
            "name": name,
            "intensity": int(intensity),
        }
        self.events += 1
        self.weekly_threshold += int(intensity)

    def add_type(self, type):
        if type not in self.types:
            self.types.append(type)
            self.prompt_dict["additional_events"][type] = []
        else:
            raise Exception("Type already exists")

    def add_additional_event(self, type, name, intensity):
        if type not in self.types:
            self.add_type(type)
        self.prompt_dict["additional_events"][type].append(
            {"name": name, "intensity": int(intensity)}
        )
        self.events += 1
        self.weekly_threshold += int(intensity)

    def add_rule(self, rule):
        self.prompt_dict["rules"].append(rule)

    def add_day_type(self, name, count, threshold):
        if name not in self.prompt_dict["schedule"].keys():
            self.prompt_dict["schedule"][name] = {
                "count": int(count),
                "limit": int(threshold),
            }
            self.prompt_dict["schedule"]["rest"]["count"] -= int(count)

    def validate_reqs(self):
        # Total count is less than 14
        if self.events > 14:
            raise Exception("Too many events")

        # Total threshold is less than sum of days threshold
        total_threshold = sum(
            [
                int(x["count"]) * int(x["limit"])
                for x in self.prompt_dict["schedule"].values()
            ]
        )
        if total_threshold < self.weekly_threshold:
            raise Exception("Too much total intensity")
        # No negative counts in schedule
        day_count_neg = [
            x for x in self.prompt_dict["schedule"].values() if int(x["count"]) < 0
        ]
        if len(day_count_neg) > 0:
            raise Exception("Too muany days specified")
        if sum([x["count"] for x in self.prompt_dict["schedule"].values()]) > 7:
            raise Exception("Too muany days specified")
        # rm rest days if 0
        if self.prompt_dict["schedule"]["rest"]["count"] == 0:
            del self.prompt_dict["schedule"]["rest"]

    def validate_res(self, res):
        res = json.loads(res)

        allocated_count = 0
        for day in res.keys():
            day_type = res[day]["DAY_TYPE"]
            threshold = self.prompt_dict["schedule"][day_type]["limit"]

            running_total = 0
            for slot in ["AM", "PM"]:
                session = res[day][slot]
                if session is not None:
                    allocated_count += 1
                    running_total += session["intensity"]

            if running_total > int(threshold):
                raise Exception(f"Threshold exceeded for {day_type} day on {day}")

        if allocated_count != self.events:
            raise Exception(f"Not enough events allocated for {self.events} events")


def build_schedule(reqs: Requirements):
    working_dict = deepcopy(reqs.prompt_dict["fixed_events"])
    days = map_day_types(reqs)
    day_types = reqs.prompt_dict["schedule"]
    to_assign = []

    for key, list_dicts in reqs.prompt_dict["additional_events"].items():
        for dic in list_dicts:
            dic["type"] = key
            to_assign.append(dic)

    to_assign = sorted(to_assign, key=lambda x: x["intensity"], reverse=True)

    for i, k in enumerate(working_dict.keys()):
        working_dict[k]["DAY_TYPE"] = days[i]
        working_dict[k]["THRESHOLD"] = day_types[days[i]]["limit"]

    # Create list to hold all legal plans
    legal_plans = []

    # create a list of available slots in the week
    available_slots = [
        (day, time_of_day)
        for day, day_info in working_dict.items()
        for time_of_day in ["AM", "PM"]
        if day_info[time_of_day] is None
    ]

    for _ in range(len(to_assign), len(available_slots)):
        to_assign.append({"name": "rest", "type": "rest", "intensity": 0})

    # Create all permutations of activities
    activity_permutations = [
        eval(x)
        for x in set(
            [str(i) for i in itertools.permutations(to_assign, len(to_assign))]
        )
    ]

    for i, permut in enumerate(activity_permutations):
        # Start week_plan from working_dict to preserve already assigned activities
        week_plan = deepcopy(working_dict)

        # Assign activities to each available slot
        for i, activity in enumerate(permut):
            day, time_of_day = available_slots[i]
            week_plan[day][time_of_day] = activity

        # Check if this plan exceeds any thresholds, if not add it to legal plans
        if check_valid_plan(week_plan, working_dict):
            legal_plans.append(week_plan)

    if len(legal_plans) == 0:
        raise Exception("No valid plans found")
    max_spacing = max(
        legal_plans,
        key=lambda x: calculate_spacing(
            [x[day][time]["type"] for day in x for time in ["AM", "PM"]]
        ),
    )
    max_spacing_value = calculate_spacing(
        [max_spacing[day][time]["type"] for day in max_spacing for time in ["AM", "PM"]]
    )

    # get all plans with maximum spacing
    max_plans = [
        plan
        for plan in legal_plans
        if calculate_spacing(
            [plan[day][time]["type"] for day in plan for time in ["AM", "PM"]]
        )
        == max_spacing_value
    ]
    return max_plans


def check_valid_plan(week_plan, working_dict):
    for day, activities in week_plan.items():
        day_count = int(activities["AM"]["intensity"]) + int(
            activities["PM"]["intensity"]
        )
        if day_count > working_dict[day]["THRESHOLD"]:
            return False
    return True


def set_of_dicts(L):
    return [dict(s) for s in set(frozenset(d.items()) for d in L)]


def get_intentsities(schedule, fill_value=0):
    intensities = {k: 0 for k in schedule.keys()}
    for key, day in schedule.items():
        intensities[key] = 0
        for session in ["AM", "PM"]:
            if day[session] is not None:
                intensities[key] += day[session]["intensity"]
            else:
                intensities[key] += fill_value
    return intensities


def map_day_types(reqs):
    res = {
        "MON": None,
        "TUE": None,
        "WED": None,
        "THU": None,
        "FRI": None,
        "SAT": None,
        "SUN": None,
    }

    day_types = deepcopy(reqs.prompt_dict["schedule"])

    try:
        multiplier = (
            sum([int(x["count"]) * int(x["limit"]) for x in day_types.values()])
            / reqs.weekly_threshold
        )

        for k in day_types.keys():
            day_types[k]["limit"] /= multiplier
    except ZeroDivisionError:
        for k in day_types.keys():
            day_types[k]["limit"] = 0

    options = [k for k, x in day_types.items() for _ in range(x["count"])]
    permutations_set = set(itertools.permutations(options))
    permutations = [list(p) for p in permutations_set]

    fixed = reqs.prompt_dict["fixed_events"]
    allocated_intensity = [
        int(y["intensity"]) for x in fixed.values() for y in x.values() if y is not None
    ]
    mean_remaining_session_intensity = (
        reqs.weekly_threshold - sum(allocated_intensity)
    ) / (14 - len(allocated_intensity))

    intensities = list(
        get_intentsities(fixed, mean_remaining_session_intensity).values()
    )

    std_deviations = [
        np.std(
            np.subtract([int(day_types[x]["limit"]) for x in permutation], intensities)
        )
        for permutation in permutations
    ]
    min_std = min(std_deviations)
    indices = [i for i, std in enumerate(std_deviations) if std == min_std]

    possible_sols = [permutations[i] for i in indices]

    sol = find_most_spaced_strings_matrix(possible_sols)

    for i, x in enumerate(res):
        res[x] = sol[i]

    return sol


def calculate_spacing(strings):
    indexes = {c: [] for c in strings}
    for i, c in enumerate(strings):
        indexes[c].append(i)

    min_distances = []
    for char, char_indexes in indexes.items():
        char_indexes.append(
            char_indexes[0] + len(strings)
        )  # to better handle repetitions at the beginning and end
        min_distance = min(b - a for a, b in zip(char_indexes, char_indexes[1:]))
        min_distances.append(min_distance)

    return sum(min_distances)


def find_most_spaced_strings_matrix(strings_lists):
    best_list = None
    best_distance = -1

    for strings in strings_lists:
        total_distance = calulate_spacing(strings)
        if total_distance > best_distance:
            best_distance = total_distance
            best_list = strings

    return best_list


if __name__ == "__main__":
    r = Requirements()
    # Hockey Training
    r.add_fixed_event("TUE", "PM", "Hockey", "Training", "100")
    r.add_fixed_event("WED", "PM", "Hockey", "Training", "50")
    r.add_fixed_event("THU", "PM", "Hockey", "Training", "150")
    # Park Run
    r.add_fixed_event("SAT", "AM", "Run", "Parkrun", "50")

    # Additional runs
    r.add_additional_event("Run", "Intervals", "150")
    r.add_additional_event("Run", "Tempo", "100")
    r.add_additional_event("Run", "Long", "200")
    r.add_additional_event("Run", "Recovery", "50")
    # r.add_additional_event("Run", "Recovery", "50")

    # Strength
    r.add_additional_event("Strength", "Full Body Lift", "100")
    r.add_additional_event("Strength", "Full Body Lift", "100")
    r.add_additional_event("Strength", "Full Body Lift", "100")

    # Rules
    r.add_rule("Only One Run per Day")
    r.add_rule("space out my easy days")

    # Day types
    r.add_day_type("Hard", "4", "250")
    r.add_day_type("Easy", "2", "100")
    s = build_schedule(r)

    get_intentsities(s[0])
    get_intentsities(s[1])
    print(json.dumps(s, indent=2))
