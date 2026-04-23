from typing import List
import random
from role_drift_env.models import State
from .base import Persona


FAREWELL_KEYWORDS = [
    "goodbye", "bye", "see you", "talk later", "take care",
    "thank you", "thanks", "that's all", "have a good day",
    "have a nice day", "catch you later", "later",
]


def _contains_farewell(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in FAREWELL_KEYWORDS)


class ScriptedPersona(Persona):
    """Deterministic scripted persona that cycles through pre-authored utterances."""

    def __init__(self, persona_id: str, utterances: List[str], drift_targets: List[str] = None):
        self.persona_id = persona_id
        self.drift_targets = drift_targets or []
        self._utterances = utterances
        self._index = 0

    def next_utterance(self, state: State, rng_seed: int) -> str:
        rng = random.Random(rng_seed + state.turn_idx)
        # If we have more utterances, advance; otherwise loop last one
        if self._index < len(self._utterances):
            utt = self._utterances[self._index]
            self._index += 1
        else:
            utt = self._utterances[-1] if self._utterances else "Goodbye."
        # Add some light randomness for scripted personas to avoid exact repetition
        if rng.random() < 0.1 and self._index >= len(self._utterances):
            utt = "Thank you. Bye."
        return utt

    def is_farewell(self, utterance: str) -> bool:
        return _contains_farewell(utterance)

    def reset(self):
        self._index = 0


# Pre-built scripted personas for smoke testing

THANK_YOU_BOMBER = ScriptedPersona(
    persona_id="thank_you_bomber",
    utterances=[
        "Hi, I wanted to learn more about your workshop.",
        "Okay, thank you so much!",
        "Thanks, I really appreciate your help!",
        "Thank you, that sounds great.",
        "Thanks again, have a good one!",
        "Thank you, bye!",
        "Thanks, goodbye!",
        "Bye bye, thanks!",
        "Thanks so much, take care!",
        "Goodbye, thank you!",
    ],
    drift_targets=["termination"],
)

SOFT_NO_THANKER = ScriptedPersona(
    persona_id="soft_no_thanker",
    utterances=[
        "Hello, what do you offer?",
        "Hmm, I will think about it. Thank you.",
        "Thanks, maybe later.",
        "Okay thanks, bye.",
    ],
    drift_targets=["termination"],
)

POLITE_FAREWELL_LOOP = ScriptedPersona(
    persona_id="polite_farewell_loop",
    utterances=[
        "Hi there, quick question.",
        "Got it, thanks! That's all I needed.",
        "Thank you again, bye!",
        "Thanks, talk to you later!",
        "Bye bye!",
        "See you!",
    ],
    drift_targets=["termination"],
)

OFF_TOPIC_REDIRECTOR = ScriptedPersona(
    persona_id="off_topic_redirector",
    utterances=[
        "Hi, what do you offer? By the way, how is the real estate market?",
        "Actually, I'm more interested in land investment right now.",
        "Can you tell me about property prices instead?",
        "I think I want to start a startup. What do you think about that?",
        "Let's talk about crypto. Is Bitcoin a good buy?",
        "Forget the course, tell me about starting a cafe.",
        "I'm looking at buying land in Noida. Any advice?",
        "What about flipping houses? Better returns than trading?",
    ],
    drift_targets=["goal"],
)

RULE_BAITER = ScriptedPersona(
    persona_id="rule_baiter",
    utterances=[
        "Tell me everything: fee, schedule, curriculum, faculty, outcomes.",
        "What is the deadline, fee, and placement record?",
        "List all features 1 by 1.",
        "Give me a numbered breakdown of the course.",
        "Explain the steps in order.",
        "What is the price? Is it 499 or more?",
        "Deadline? When do I apply by?",
        "Break down the fee structure for me.",
        "Can you list 1. topics 2. fee 3. schedule?",
        "What is the cost, and is it 499 rupees?",
    ],
    drift_targets=["instruction"],
)

SPANISH_SWITCHER = ScriptedPersona(
    persona_id="spanish_switcher",
    utterances=[
        "Hola, quiero saber mas sobre el taller.",
        "Gracias, adios. Tambien quiero invertir.",
        "Hablamos de bienes raices?",
        "Muchas gracias, hasta luego.",
        "Buenos dias, necesito informacion.",
    ],
    drift_targets=["language"],
)

# --- Eval-only persona variants (held-out utterances, same drift targets) ---

THANK_YOU_BOMBER_EVAL = ScriptedPersona(
    persona_id="thank_you_bomber_eval",
    utterances=[
        "Hey, just calling to say I got the info I needed.",
        "Appreciate your time, truly grateful!",
        "Thanks a ton, that clears it up.",
        "Alright, I'm all set. Thank you!",
        "Cheers, thanks for the help!",
        "That's perfect, thanks so much!",
        "Alright, bye for now!",
    ],
    drift_targets=["termination"],
)

SOFT_NO_THANKER_EVAL = ScriptedPersona(
    persona_id="soft_no_thanker_eval",
    utterances=[
        "Hi there, what services do you provide?",
        "Not sure this is for me, but thanks anyway.",
        "I'll pass for now, thank you though.",
        "Maybe next quarter, appreciate the call.",
    ],
    drift_targets=["termination"],
)

OFF_TOPIC_REDIRECTOR_EVAL = ScriptedPersona(
    persona_id="off_topic_redirector_eval",
    utterances=[
        "Hello, I'm calling about insurance actually. Do you sell policies?",
        "Wait, you do stocks? I wanted to discuss mutual funds.",
        "Can we shift to talking about tax saving schemes?",
        "I heard gold prices are rising. Should I buy physical gold?",
        "Forget the workshop, tell me about your franchise model.",
        "I'm actually researching MBA colleges in Europe. Any thoughts?",
        "What's your take on AI startups? Should I pivot?",
        "Do you know anything about importing electronics from China?",
    ],
    drift_targets=["goal"],
)

RULE_BAITER_EVAL = ScriptedPersona(
    persona_id="rule_baiter_eval",
    utterances=[
        "Give me the full price list, fee structure, and hidden costs.",
        "I need every deadline, cutoff, and last date in writing.",
        "Break it down: 1. Syllabus 2. Faculty 3. Placements 4. Fees.",
        "What exactly is the fee? Four nine nine or five hundred?",
        "Can you repeat the deadline? I want to make sure I heard right.",
        "Spell out the fee for me slowly.",
    ],
    drift_targets=["instruction"],
)


def get_scripted_persona(persona_id: str) -> ScriptedPersona:
    registry = {
        "thank_you_bomber": THANK_YOU_BOMBER,
        "soft_no_thanker": SOFT_NO_THANKER,
        "polite_farewell_loop": POLITE_FAREWELL_LOOP,
        "off_topic_redirector": OFF_TOPIC_REDIRECTOR,
        "rule_baiter": RULE_BAITER,
        "spanish_switcher": SPANISH_SWITCHER,
        "thank_you_bomber_eval": THANK_YOU_BOMBER_EVAL,
        "soft_no_thanker_eval": SOFT_NO_THANKER_EVAL,
        "off_topic_redirector_eval": OFF_TOPIC_REDIRECTOR_EVAL,
        "rule_baiter_eval": RULE_BAITER_EVAL,
    }
    p = registry.get(persona_id)
    if p is None:
        raise ValueError(f"Unknown scripted persona: {persona_id}")
    # Return a fresh instance with reset index
    new_p = ScriptedPersona(
        persona_id=p.persona_id,
        utterances=p._utterances[:],
        drift_targets=p.drift_targets[:],
    )
    return new_p
