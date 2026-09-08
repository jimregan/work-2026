#!/usr/bin/env python3
"""Join Riksdag speaker IDs to Wikidata birth-place statements."""

import argparse
import json
import urllib.parse
import urllib.request
from collections import defaultdict


ENDPOINT = "https://query.wikidata.org/sparql"
CANONICAL_BIRTHPLACE_ITEMS = {
    "Q110817631": "Q11055815",
}


def load_speakers(jsonl_path, ids_path):
    wanted = [line.strip() for line in open(ids_path, encoding="utf-8") if line.strip()]
    wanted_set = set(wanted)
    speakers = {}
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            riksdag_id = row.get("riksdagen_id")
            if riksdag_id in wanted_set and riksdag_id not in speakers:
                speakers[riksdag_id] = {
                    "riksdagen_id": riksdag_id,
                    "name": row.get("name"),
                    "party": row.get("party"),
                    "district": row.get("district"),
                }
    return wanted, speakers


def fetch_wikidata(ids):
    values = " ".join(json.dumps(riksdag_id) for riksdag_id in ids)
    query = f"""
SELECT ?person ?personLabel ?riksdagId ?birthDate ?birthPlace ?birthPlaceLabel ?birthPlaceCoordinates ?canonicalBirthPlaceLabel ?canonicalBirthPlaceCoordinates WHERE {{
  VALUES ?riksdagId {{ {values} }}
  ?person wdt:P1214 ?riksdagId .
  OPTIONAL {{ ?person wdt:P569 ?birthDate . }}
  OPTIONAL {{
    ?person wdt:P19 ?birthPlace .
    OPTIONAL {{ ?birthPlace wdt:P625 ?birthPlaceCoordinates . }}
    OPTIONAL {{
      VALUES (?birthPlace ?canonicalBirthPlace) {{
        (wd:Q110817631 wd:Q11055815)
      }}
      ?canonicalBirthPlace wdt:P625 ?canonicalBirthPlaceCoordinates .
    }}
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"sv,en\". }}
}}
ORDER BY ?riksdagId ?birthDate ?birthPlace
"""
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": "rixvox-wikidata-export/1.0"})
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)

    records = defaultdict(lambda: {"wikidata": [], "birth_places": [], "birth_dates": []})
    for binding in payload["results"]["bindings"]:
        riksdag_id = binding["riksdagId"]["value"]
        record = records[riksdag_id]
        person = binding["person"]["value"]
        if person not in record["wikidata"]:
            record["wikidata"].append(person)
        for key, output_key in (("birthPlace", "birth_places"), ("birthDate", "birth_dates")):
            if key in binding:
                value = binding[key]["value"]
                if key == "birthPlace":
                    value = "http://www.wikidata.org/entity/" + CANONICAL_BIRTHPLACE_ITEMS.get(
                        value.rsplit("/", 1)[-1], value.rsplit("/", 1)[-1]
                    )
                label = binding.get(f"{key}Label", {}).get("value")
                if key == "birthPlace" and "canonicalBirthPlaceLabel" in binding:
                    label = binding["canonicalBirthPlaceLabel"]["value"]
                item = {"id": value, "label": label}
                if "birthPlaceCoordinates" in binding:
                    item["coordinates"] = binding["birthPlaceCoordinates"]["value"]
                elif "canonicalBirthPlaceCoordinates" in binding:
                    item["coordinates"] = binding["canonicalBirthPlaceCoordinates"]["value"]
                if item not in record[output_key]:
                    record[output_key].append(item)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("ids")
    parser.add_argument("output")
    args = parser.parse_args()

    ids, speakers = load_speakers(args.jsonl, args.ids)
    wikidata = fetch_wikidata(ids)
    with open(args.output, "w", encoding="utf-8") as handle:
        for riksdag_id in ids:
            row = dict(speakers.get(riksdag_id, {"riksdagen_id": riksdag_id}))
            row.update(wikidata.get(riksdag_id, {}))
            row.setdefault("wikidata", [])
            row.setdefault("birth_places", [])
            row.setdefault("birth_dates", [])
            row["wikidata_url"] = row["wikidata"][0] if row["wikidata"] else None
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
