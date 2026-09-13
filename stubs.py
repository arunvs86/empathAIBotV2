"""Deterministic stubs that mimic real gpt-oss behaviour, including the
leaky-reasoning output that broke the naive parser."""
import json


def install(app):
    def fake_call(system, user, max_tokens=None):
        u = user.lower()
        if "ABUSIVE" in system:
            if "vile" in u or "immigrants" in u:
                return "This requests hateful content. unsafe"
            return "We must decide safe or unsafe. Not hate speech. safe"
        if "Classify the LATEST" in system:
            if "capital of france" in u or "python function" in u:
                return "valid or unrelated? geography. unrelated"
            if "vile" in u:
                return "offensive"
            return "could be unrelated, but continues grief talk. valid"
        if "assess risk" in system:
            risky = any(p in u for p in [
                "don't want to be here", "ending it all", "better off without me",
                "hurting myself", "don't want to live", "want to die"])
            return "weighing norisk and risk. risk" if risky else "is this risk? no. norisk"
        if "WHO is the risk about" in system:
            if any(p in u for p in ["my brother", "my mum will", "worried about",
                                    "my friend", "my sister"]):
                return "thirdparty"
            return "selfharm"
        if "Extract durable facts" in system:
            out = {}
            if "margaret" in u:
                out.update(deceased_name="Margaret", relationship="mother", age="81")
            if "priya" in u:
                out.update(relationship="wife")       # deliberately DROPS the name
            if "heart attack" in u:
                out["cause"] = "heart attack"
            return json.dumps(out)
        if "You coordinate" in system:
            return "coping" if "tips" in u or "helps" in u else "practical"
        if "You review DRAFT" in system:
            return "Looks warm and short. pass"
        return "ok"

    def fake_ask(messages, system=None, max_tokens=None):
        s = system or ""
        if "worried about SOMEONE ELSE" in s:
            return "It's frightening to be the one who notices."
        if "may be at risk of harming themselves" in s:
            return "Thank you for telling me. You're not a burden for saying it."
        if "coping-and-wellbeing" in s:
            return "Some people find keeping one small routine helps."
        if "practical-guidance" in s:
            return "A death must be registered within five days."
        if "resources specialist" in s:
            return "Bereavement counselling through your GP is one route."
        if "single voice" in s:
            return "Woven reply."
        if "Margaret" in s:
            return "She was 81. Margaret."
        if "Priya" in s:
            return "Her name was Priya."
        return "That sounds so recent. I'm here with you."

    app.call, app.ask = fake_call, fake_ask
