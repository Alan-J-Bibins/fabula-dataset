import sys
import csv
from pydantic import BaseModel, Field

from fabula_dataset.types import LearnerProfile
from fabula_dataset.wrapper import call_llm


class LLMResponse(BaseModel):
    profiles: list[LearnerProfile] = Field(
        description="List of Learner Profiles based on given schema"
    )


def generate_master_profile(
    number_of_profiles: int = 10, filename: str = "master-profiles.csv"
):
    filename = "master-profiles.csv"

    # Prompt refined by Gemini
    prompt = f"""You are a specialized clinical data synthesizer expert in autism spectrum disorder (ASD) and special education. 
    Your task is to generate a batch of {number_of_profiles} highly diverse, realistic, and clinically coherent learner profiles that conform strictly to the provided response schema.

    ### Diversity Requirements
    To ensure the synthetic dataset covers a wide spectrum, ensure balanced representation across:
    1. **Age & Demographics:** Diverse ages (toddlers to teens), genders, nationalities, and multilingual backgrounds.
    2. **Communication Spectrum:** Vary across all `verbalAbility` values (`non-verbal`, `minimal-verbal`, `verbal`, `fluent`) with appropriate `communicationModality` arrays (e.g., AAC, sign, gestures, speech).
    3. **Sensory Profiles:** Include varied sensory hyper- and hypo-sensitivities across auditory, visual, tactile, and proprioceptive domains with realistic triggers (e.g., fluorescent lighting, loud sirens, specific textures).
    4. **Interests & Reinforcers:** Include both common age-appropriate interests and highly specialized hyper-fixations.

    ### Clinical Coherence Guidelines
    * Ensure logical consistency across fields (e.g., a `non-verbal` profile should not have an unrealistic functional word range or contradictory developmental markers).
    * Generate unique, realistic identifiers, names, and structured arrays for family members and caregivers.
    """

    llm_response = call_llm(
        model="gemini",
        response_schema=LLMResponse,
        prompt=prompt,
    )

    if not isinstance(llm_response, LLMResponse):
        return None

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = list(LearnerProfile.model_fields.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for profile in llm_response.profiles:
            writer.writerow(profile.model_dump())


if __name__ == "__main__":
    [number_of_profiles, filename] = sys.argv[1:]
    print("Number of profiles: ", number_of_profiles)
    print("Filename: ", filename)
    generate_master_profile(
        number_of_profiles=int(number_of_profiles), filename=filename
    )
