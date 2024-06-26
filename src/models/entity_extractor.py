import spacy

class SpacyEntityExtractor:
    def __init__(self, model_name='en_core_web_sm'):
        self.nlp = spacy.load(model_name)

    def extract_entities(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

# Entity types in Spacy
# PERSON - People, including fictional.
# NORP - Nationalities or religious or political groups.
# FAC - Buildings, airports, highways, bridges, etc.
# ORG - Companies, agencies, institutions, etc.
# GPE - Countries, cities, states.
# LOC - Non-GPE locations, mountain ranges, bodies of water.
# PRODUCT - Objects, vehicles, foods, etc. (Not services.)
# EVENT - Named hurricanes, battles, wars, sports events, etc.
# WORK_OF_ART - Titles of books, songs, etc.
# LAW - Named documents made into laws.
# LANGUAGE - Any named language.
# DATE - Absolute or relative dates or periods.
# TIME - Times smaller than a day.
# PERCENT - Percentage, including "%".
# MONEY - Monetary values, including unit.
# QUANTITY - Measurements, as of weight or distance.
# ORDINAL - "first", "second", etc.
# CARDINAL - Numerals that do not fall under another type.