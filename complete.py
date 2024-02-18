#! python3
from tabulate import tabulate
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


def build_schedule(reqs: Requirements, fitness=None, starting_fatigue=0):

    if fitness is None:
        fitness = reqs.weekly_threshold / 7

    working_dict = deepcopy(reqs.prompt_dict["fixed_events"])
    to_assign = []

    for key, list_dicts in reqs.prompt_dict["additional_events"].items():
        for dic in list_dicts:
            dic["type"] = key
            to_assign.append(dic)

    to_assign = sorted(to_assign, key=lambda x: x["intensity"], reverse=True)

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
        if check_valid_plan(week_plan, fitness, starting_fatigue):
            legal_plans.append(week_plan)

    if len(legal_plans) == 0:
        raise Exception("No valid plans found")
    # We then want to maximise the spacing of intensity from each type
    best_plan = max(legal_plans, key=lambda x: plan_score(x))
    return best_plan


def by_type_recovery_scores(intensity_dict):
    scores = {}
    for workout, intensities in intensity_dict.items():
        scores[workout] = calc_recovery_score(intensities)

    return scores


def calc_recovery_score(intensities: list, recovery_value=None, starting_fatigue=0):
    # Each Session I recover 1/14 of the total intensity
    if not recovery_value:
        recovery_value = 1 / 14 * sum(intensities)
    fatigue_list = [-starting_fatigue] * 14
    for i, x in enumerate(intensities):
        fatigue_list[i] = fatigue_list[i - 1] - x + recovery_value
    return fatigue_list


def overall_recovery_score(intensity_dict, fitness, starting_fatigue):
    intensity_list = sum(np.array(list(intensity_dict.values())))
    recovery_score = calc_recovery_score(intensity_list, fitness / 2, starting_fatigue)
    return recovery_score


def check_valid_plan(plan, fitness, starting_fatigue):
    intensity_dict = calculate_intensity_dict(plan)
    recovery_score = overall_recovery_score(intensity_dict, fitness, starting_fatigue)
    if min(recovery_score) < -fitness:
        return False
    else:
        return True


def calculate_intensity_dict(plan):
    # initialize intensity_dict with the total intensity
    intensity_dict = {}

    # Iterate over the days of week
    for i, day in enumerate(plan.values()):
        for j, session in enumerate(["AM", "PM"]):
            if day[session]["type"] not in intensity_dict.keys():
                intensity_dict[day[session]["type"]] = [0] * 14
            # Add the intensities to their respective lists
            intensity_dict[day[session]["type"]][2 * i + j] += day[session]["intensity"]
    return intensity_dict


def plan_score(plan, return_dict=False):
    # initialize intensity_dict with the total intensity
    intensity_dict = calculate_intensity_dict(plan)
    scores = by_type_recovery_scores(intensity_dict)
    if return_dict:
        return scores
    else:
        return sum(min(scores[x]) for x in scores.keys())


def pretty_print_table(schedule):
    # Create a list of time slots
    time_slots = ["AM", "PM"]

    # Retrieve the days from the dictionary
    days = list(schedule.keys())

    # Create a list to store the rows of the table
    rows = []

    # Create a list for the day headers, including type
    day_headers = [f"{day}" for day in days]

    # Iterate over the time slots
    for time_slot in time_slots:
        # Create a row for the current time slot
        row = [time_slot]

        # Iterate over the days of the week
        for day in days:
            # Get the activity for the current day and time slot
            activity = schedule[day].get(time_slot, "N/A")
            # Create a string representation of the activity
            activity_str = (
                f'{activity["type"]} {activity["name"]} ({activity["intensity"]})'
            )
            # Add the string representation of the activity to the row
            row.append(activity_str)

        # Add the row to the table rows
        rows.append(row)

    # Print the table using tabulate
    print(tabulate(rows, headers=["Time slot"] + day_headers))


if __name__ == "__main__":
    r = Requirements()
    # Hockey Training
    r.add_fixed_event("TUE", "PM", "Hockey", "Training", "150")
    r.add_fixed_event("WED", "PM", "Hockey", "Training", "80")
    r.add_fixed_event("THU", "PM", "Hockey", "Training", "150")
    r.add_fixed_event("SUN", "PM", "Hockey", "Game", "100")
    # Park Run
    r.add_fixed_event("SAT", "AM", "Run", "Parkrun", "50")

    # Additional runs
    r.add_additional_event("Run", "Intervals", "150")
    r.add_additional_event("Run", "Tempo", "100")
    r.add_additional_event("Run", "Long", "200")
    r.add_additional_event("Run", "Recovery", "50")
    r.add_additional_event("Run", "Recovery", "50")

    # Strength
    r.add_additional_event("Strength", "Full Body Lift", "30")
    r.add_additional_event("Strength", "Full Body Lift", "30")
    r.add_additional_event("Strength", "Full Body Lift", "30")

    # Rules
    r.add_rule("Only One Run per Day")

    # Day types
    s = build_schedule(r, fitness=170, starting_fatigue=0)
    f = plan_score(s, True)
    training_load = r.weekly_threshold / 7
    total_fatigue = sum(np.array(x) for x in f.values())

    print(json.dumps(f, indent=2))
    pretty_print_table(s)
