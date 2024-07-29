# THIS METHOD OF COMMAND PROCESSING IS NO LONGER USED, IT IS KEPT IN THE REPOSITORY AS LEGACY FOR NOW
import spacy

class ClauseExtractor:
    def __init__(self, model_name='en_core_web_sm'):
        self.nlp = spacy.load(model_name)

    def extract_clauses(self, text):
        """
        Extracts complement and embedded clauses from the given text and identifies the dependency label.
        
        Args:
        text (str): The input text from which to extract clauses.

        Returns:
        list of tuples: A list of tuples where each tuple contains a clause and its corresponding dependency label.
        """
        doc = self.nlp(text)
        clauses = []
        for sent in doc.sents:
            for token in sent:
                # if token.dep_ in ("ccomp", "conj", "xcomp", "acl", "advcl", "relcl"):  # Complement clauses
                if token.dep_ in ("ccomp", "conj", "xcomp", "advcl", "relcl", "pobj"):  # Complement clauses
                    clause = " ".join([child.text for child in token.subtree])
                    clauses.append((clause, token.dep_))
        return clauses
