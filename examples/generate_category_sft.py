from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CATEGORIES = {
    "APPAREL": [
        "women's cotton crew neck t-shirt size {size}",
        "men's slim fit denim jeans waist {waist}",
        "kids waterproof hooded rain jacket size {size}",
    ],
    "ELECTRONICS": [
        "wireless bluetooth earbuds with charging case",
        "65W USB-C laptop wall charger",
        "mechanical gaming keyboard with hot-swappable switches",
    ],
    "HOME_KITCHEN": [
        "stainless steel frying pan {inch} inch",
        "memory foam bed pillow standard size",
        "ceramic coffee mug {oz} oz dishwasher safe",
    ],
    "BEAUTY": [
        "hydrating facial moisturizer with hyaluronic acid",
        "volumizing mascara waterproof black",
        "sulfate-free shampoo for dry hair",
    ],
    "GROCERY": [
        "organic roasted almonds {oz} oz bag",
        "whole bean medium roast coffee {oz} oz",
        "sparkling mineral water pack of {count}",
    ],
    "SPORTS": [
        "adjustable dumbbell set for home gym",
        "yoga mat non-slip {mm} mm thick",
        "outdoor soccer ball size {size}",
    ],
    "AUTOMOTIVE": [
        "windshield wiper blades pair {inch} inch",
        "synthetic motor oil 5W-30 five quart",
        "portable tire inflator with digital gauge",
    ],
    "BOOKS": [
        "paperback introduction to machine learning",
        "hardcover children's illustrated story book",
        "spiral bound weekly planner 2027",
    ],
}

SYSTEM = (
    "Classify the product into exactly one category. "
    "Return only the category label."
)


def render(template: str, rng: random.Random) -> str:
    return template.format(
        size=rng.choice(["S", "M", "L", "XL", "5"]),
        waist=rng.choice(["30", "32", "34", "36"]),
        inch=rng.choice(["8", "10", "12", "20", "24"]),
        oz=rng.choice(["8", "12", "16", "24", "32"]),
        count=rng.choice(["6", "12", "24"]),
        mm=rng.choice(["4", "6", "8"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    labels = tuple(CATEGORIES)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for index in range(args.rows):
            label = labels[index % len(labels)]
            title = render(rng.choice(CATEGORIES[label]), rng)
            record = {
                "id": index,
                "system": SYSTEM,
                "user": f"Product: {title}",
                "assistant": label,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
