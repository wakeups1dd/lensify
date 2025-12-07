# Example usage (upgraded Lensify)

# Install recommended deps:
pip install -r requirements.txt
# Download NLTK punkt if needed:
python -c "import nltk; nltk.download('punkt')"

# Rebuild index
lensify rebuild ./docs --show_progress

# Query
lensify query ./docs "payment risk simulation" --k 5

# Stats
lensify stats ./docs

# Doctor
lensify doctor

# Export
lensify export ./docs out.json
