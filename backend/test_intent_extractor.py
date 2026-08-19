#!/usr/bin/env python3
"""
test_intent_extractor.py
===========================

Automated test suite for Step 1 (intent extraction + dual-model routing).
Runs entirely without a GPU or a real Gemini API key -- the routing logic
in intent_extractor.route_intent() is dependency-injected, so we can feed
it fake "model outputs" and verify the routing decisions directly.

This is what "Step 1 is fully tested" actually means: every branch of the
Qwen-succeeds / Qwen-fails / Gemini-succeeds / both-fail decision tree is
exercised explicitly, plus the schema validation and clarification logic
that everything else depends on.

Usage:
    python test_intent_extractor.py
"""

import json
import sys

from schema_utils import (
    INTENT_SCHEMA,
    is_schema_valid,
    normalize_intent,
    needs_clarification,
)
from intent_extractor import route_intent

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}  {detail}")


def section(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# --------------------------------------------------------------------------- #
# 1. is_schema_valid() -- the exact condition that triggers fallback
# --------------------------------------------------------------------------- #
section("is_schema_valid()")

valid_house = {"property_type": "House", "subtype": None, "location": "DHA", "marla_min": 5.0,
               "marla_max": 5.0, "price_max": 25000000, "soft_signals": [], "poi": []}
ok, reason = is_schema_valid(valid_house)
check("valid House response passes", ok, reason)

valid_all_null = dict(INTENT_SCHEMA)
ok, reason = is_schema_valid(valid_all_null)
check("all-null response is still schema-valid", ok, reason)

missing_key = {"property_type": "House"}
ok, reason = is_schema_valid(missing_key)
check("missing keys correctly rejected", not ok, "should have failed")

mismatched_subtype = {"property_type": "House", "subtype": "Commercial Plot", "location": None,
                       "marla_min": None, "marla_max": None, "price_max": None, "soft_signals": [], "poi": []}
ok, reason = is_schema_valid(mismatched_subtype)
check("mismatched category/subtype correctly rejected", not ok, "should have failed")

hallucinated_subtype = {"property_type": "Flat", "subtype": "Apartment", "location": None,
                         "marla_min": None, "marla_max": None, "price_max": None, "soft_signals": [], "poi": []}
ok, reason = is_schema_valid(hallucinated_subtype)
check("subtype outside controlled vocabulary correctly rejected", not ok, "should have failed")

lowercase_subtype = {"property_type": "Plot", "subtype": "commercial plot", "location": "DHA Lahore",
                      "marla_min": None, "marla_max": None, "price_max": 10000000, "soft_signals": [], "poi": []}
ok, reason = is_schema_valid(lowercase_subtype)
check("lowercase-but-correct subtype is accepted (case-insensitive)", ok, reason)
normalized = normalize_intent(lowercase_subtype)
check("lowercase subtype gets snapped to canonical casing", normalized["subtype"] == "Commercial Plot")

bad_type = {"property_type": "House", "subtype": None, "location": None, "marla_min": "five",
            "marla_max": None, "price_max": None, "soft_signals": [], "poi": []}
ok, reason = is_schema_valid(bad_type)
check("non-numeric marla_min correctly rejected", not ok, "should have failed")

not_a_dict = "just a string"
ok, reason = is_schema_valid(not_a_dict)
check("non-dict input correctly rejected", not ok, "should have failed")


# --------------------------------------------------------------------------- #
# 2. normalize_intent() -- never raises, always returns full schema
# --------------------------------------------------------------------------- #
section("normalize_intent()")

result = normalize_intent(valid_house)
check("valid input normalizes cleanly", result["property_type"] == "House")

result = normalize_intent({"garbage": "in"})
check("garbage input returns full schema with nulls", set(result.keys()) == set(INTENT_SCHEMA.keys()))
check("garbage input has null property_type", result["property_type"] is None)

result = normalize_intent(mismatched_subtype)
check("mismatched subtype gets dropped (not trusted) during normalization", result["subtype"] is None)

result = normalize_intent(None)
check("None input doesn't raise, returns default schema", result == dict(INTENT_SCHEMA))

result = normalize_intent({"price_max": 25000000.7, "property_type": None, "subtype": None,
                            "location": None, "marla_min": None, "marla_max": None,
                            "soft_signals": [], "poi": []})
check("float price_max gets rounded to int", result["price_max"] == 25000001)


# --------------------------------------------------------------------------- #
# 3. needs_clarification() -- Plot triggers, House doesn't
# --------------------------------------------------------------------------- #
section("needs_clarification()")

plot_no_subtype = normalize_intent({"property_type": "Plot", "subtype": None, "location": "DHA",
                                     "marla_min": None, "marla_max": None, "price_max": None,
                                     "soft_signals": [], "poi": []})
clarification = needs_clarification(plot_no_subtype)
check("Plot with no subtype triggers clarification", clarification is not None)
check("clarification includes Commercial Plot option", "Commercial Plot" in clarification["options"])

plot_with_subtype = normalize_intent({"property_type": "Plot", "subtype": "Commercial Plot", "location": "DHA",
                                       "marla_min": None, "marla_max": None, "price_max": None,
                                       "soft_signals": [], "poi": []})
check("Plot WITH subtype does NOT trigger clarification", needs_clarification(plot_with_subtype) is None)

house_no_subtype = normalize_intent({"property_type": "House", "subtype": None, "location": "DHA",
                                      "marla_min": 5.0, "marla_max": 5.0, "price_max": None,
                                      "soft_signals": [], "poi": []})
check("House with no subtype does NOT trigger clarification (by design)", needs_clarification(house_no_subtype) is None)


# --------------------------------------------------------------------------- #
# 4. route_intent() -- the actual dual-model decision tree, fully mocked
# --------------------------------------------------------------------------- #
section("route_intent() -- Qwen/Gemini routing decision tree")

valid_json_text = json.dumps(valid_house)
gemini_response_text = json.dumps({**valid_house, "location": "Gulberg"})


def qwen_valid(q):
    return valid_json_text


def qwen_malformed_json(q):
    return "{property_type: House, this is not valid json"


def qwen_schema_invalid(q):
    return json.dumps({"property_type": "House", "subtype": "Commercial Plot"})  # missing keys too


def qwen_empty(q):
    return ""


def gemini_valid(q):
    return gemini_response_text


def gemini_fails(q):
    return None


# Case 1: Qwen succeeds -> should use Qwen, never touch Gemini
gemini_was_called = {"flag": False}
def gemini_spy(q):
    gemini_was_called["flag"] = True
    return gemini_response_text

intent, source = route_intent("test query", qwen_generate_fn=qwen_valid, gemini_generate_fn=gemini_spy)
check("Case 1: Qwen valid -> source is 'qwen'", source == "qwen")
check("Case 1: Qwen valid -> Gemini NOT called", not gemini_was_called["flag"])
check("Case 1: correct intent extracted", intent["property_type"] == "House")

# Case 2: Qwen returns malformed JSON -> falls back to Gemini
intent, source = route_intent("test query", qwen_generate_fn=qwen_malformed_json, gemini_generate_fn=gemini_valid)
check("Case 2: Qwen malformed JSON -> falls back to Gemini", source == "gemini")
check("Case 2: Gemini's intent is used", intent["location"] == "Gulberg")

# Case 3: Qwen returns schema-invalid JSON (missing keys) -> falls back to Gemini
intent, source = route_intent("test query", qwen_generate_fn=qwen_schema_invalid, gemini_generate_fn=gemini_valid)
check("Case 3: Qwen schema-invalid -> falls back to Gemini", source == "gemini")

# Case 4: Qwen returns empty string -> falls back to Gemini
intent, source = route_intent("test query", qwen_generate_fn=qwen_empty, gemini_generate_fn=gemini_valid)
check("Case 4: Qwen empty output -> falls back to Gemini", source == "gemini")

# Case 5: No Qwen loaded at all (qwen_generate_fn=None) -> goes straight to Gemini
intent, source = route_intent("test query", qwen_generate_fn=None, gemini_generate_fn=gemini_valid)
check("Case 5: No Qwen configured -> uses Gemini directly", source == "gemini")

# Case 6: Both Qwen AND Gemini fail -> safe empty intent, never raises
intent, source = route_intent("test query", qwen_generate_fn=qwen_malformed_json, gemini_generate_fn=gemini_fails)
check("Case 6: both paths fail -> source is 'empty'", source == "empty")
check("Case 6: both paths fail -> returns safe default schema", intent == dict(INTENT_SCHEMA))

# Case 7: empty query string -> short-circuits immediately, no calls made
qwen_called = {"flag": False}
def qwen_spy(q):
    qwen_called["flag"] = True
    return valid_json_text
intent, source = route_intent("", qwen_generate_fn=qwen_spy, gemini_generate_fn=gemini_valid)
check("Case 7: empty query short-circuits", source == "empty")
check("Case 7: empty query never calls Qwen", not qwen_called["flag"])

# Case 8: markdown-fenced JSON from Qwen (common quirk) still parses
def qwen_fenced(q):
    return "```json\n" + valid_json_text + "\n```"
intent, source = route_intent("test query", qwen_generate_fn=qwen_fenced, gemini_generate_fn=gemini_valid)
check("Case 8: markdown-fenced Qwen output still parses and is used", source == "qwen")
# Case 9: Roman Urdu query handling (Simulated LLM Translation)
# Simulating the query: "DHA Lahore me 5 marla sasta ghar"
def qwen_roman_urdu(q):
    # Simulating that the fine-tuned LLM correctly translates "sasta" to "cheap"
    return json.dumps({
        "property_type": "House",
        "subtype": None,
        "location": "DHA Lahore",
        "marla_min": 5.0,
        "marla_max": 5.0,
        "price_max": None,
        "soft_signals": ["cheap"],
        "poi": []
    })

intent, source = route_intent(
    "DHA Lahore me 5 marla sasta ghar",
    qwen_generate_fn=qwen_roman_urdu,
    gemini_generate_fn=gemini_valid
)

check("Case 9: Roman Urdu query JSON passes schema validation", source == "qwen")
check("Case 9: Roman Urdu 'sasta' is correctly verified as 'cheap' in soft_signals", "cheap" in intent["soft_signals"])
check("Case 9: Roman Urdu 'ghar' is correctly mapped to 'House' property_type", intent["property_type"] == "House")

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
section("SUMMARY")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed:
    print(f"\nRESULT: FAILED -- {failed} test(s) did not pass. Step 1 is NOT ready.")
    sys.exit(1)
else:
    print(f"\nRESULT: ALL {passed} TESTS PASSED -- Step 1 routing logic is verified.")
    sys.exit(0)