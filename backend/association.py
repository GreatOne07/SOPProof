
def associate_helmets(
    people,
    helmets,
    head_height_ratio=0.35
):
    """
    Associate helmet detections with approximate
    worker head regions.

    An unmatched person is flagged for human review,
    not automatically declared non-compliant.
    """

    candidates = []

    # Find possible worker-to-helmet matches
    for pi, person in enumerate(people):

        person_width = person["right"] - person["left"]
        person_height = person["bottom"] - person["top"]

        if person_width <= 0 or person_height <= 0:
            continue

        head_bottom = (
            person["top"] +
            head_height_ratio * person_height
        )

        head_center_x = (
            person["left"] + person["right"]
        ) / 2

        head_center_y = (
            person["top"] + head_bottom
        ) / 2

        for hi, helmet in enumerate(helmets):

            helmet_x = (
                helmet["left"] + helmet["right"]
            ) / 2

            helmet_y = (
                helmet["top"] + helmet["bottom"]
            ) / 2

            inside_head = (
                person["left"] <= helmet_x <= person["right"]
                and person["top"] <= helmet_y <= head_bottom
            )

            if inside_head:

                score = (
                    abs(helmet_x - head_center_x)
                    / person_width
                    +
                    abs(helmet_y - head_center_y)
                    / max(head_bottom - person["top"], 1)
                )

                candidates.append((score, pi, hi))

    # Assign each helmet to at most one person
    candidates.sort()

    assignments = {}
    used_helmets = set()

    for score, pi, hi in candidates:

        if pi not in assignments and hi not in used_helmets:
            assignments[pi] = hi
            used_helmets.add(hi)

    # Build inspection results
    results = []

    for pi, person in enumerate(people):

        result = {
            "person_number": pi + 1,
            "person_confidence": person["confidence"],
            "helmet": None,
            "helmet_confidence": None,
            "status": "review_required",
        }

        if pi in assignments:
            helmet = helmets[assignments[pi]]

            result.update({
                "helmet": assignments[pi] + 1,
                "helmet_confidence": helmet["confidence"],
                "status": "helmet_visible",
            })

        results.append(result)

    unmatched = sum(
        1 for item in results
        if item["status"] == "review_required"
    )

    return {
        "people_count": len(people),
        "helmet_count": len(helmets),
        "matched_count": len(people) - unmatched,
        "review_count": unmatched,
        "human_review_required": unmatched > 0,
        "overall_result": (
            "all_helmets_visible"
            if unmatched == 0 and len(people) > 0
            else "review_required"
        ),
        "people": results,
    }

