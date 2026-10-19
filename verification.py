"""Shared validation of the outputs produced by the QWEN model.

This module will contain the common logic used by every project track (structured
cohort data, questionnaire data, medical text) to decide whether an output of the
model is accepted or rejected before it is used to build ontology content.

Typical checks expected here:
- the output can be parsed into the expected structure (concepts, triplets, mappings);
- mandatory fields are present and non-empty;
- the proposed concepts and identifiers can be verified against SNOMED CT;
- each result receives a validation status: Correct / Incorrect / Uncertain;
- concepts with no SNOMED CT match are flagged as Candidate Project Concepts.

No implementation yet: the functions will be defined jointly by the three tracks.
"""
