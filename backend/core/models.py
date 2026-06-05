import logging
from typing import Optional

logger = logging.getLogger('ats_resume_scorer')

def ensure_nlp(app) -> None:
    """Lazily load spaCy model into `app.state.nlp` if not present."""
    if getattr(app.state, 'nlp', None) is None:
        try:
            import spacy
            from backend.core.config import SPACY_MODEL_PRIMARY, SPACY_MODEL_SECONDARY
            app.state.nlp = spacy.load(SPACY_MODEL_PRIMARY)
            logger.info(f'Loaded spaCy model: {SPACY_MODEL_PRIMARY}')
        except Exception:
            logger.warning('Primary spaCy model not found, trying secondary')
            try:
                import spacy
                from backend.core.config import SPACY_MODEL_SECONDARY
                app.state.nlp = spacy.load(SPACY_MODEL_SECONDARY)
                logger.info(f'Loaded spaCy model (fallback): {SPACY_MODEL_SECONDARY}')
            except Exception as exc:
                logger.error(f'Could not load any spaCy model: {exc}')
                raise

def ensure_embedder(app) -> None:
    """Lazily load SentenceTransformer into `app.state.embedder` if not present."""
    if getattr(app.state, 'embedder', None) is None:
        try:
            from sentence_transformers import SentenceTransformer
            from backend.core.config import SENTENCE_TRANSFORMER_MODEL
            # Defer heavy model loading until needed
            app.state.embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
            logger.info(f'Loaded SentenceTransformer: {SENTENCE_TRANSFORMER_MODEL}')
        except Exception as exc:
            logger.error(f'Failed to load embedder: {exc}')
            raise
