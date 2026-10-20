# MP2 Reflection

Run mp2_rag.py ask: 

Upon using 3 questions from learner_questions.jsonl - 

Q1: Whom does Holmes identify as the thief of Countess' blue carbuncle and who was falsely accused ?
Answer: Holmes identifies James Ryder as the thief of the Countess' blue carbuncle. The falsely accused individual is John Horner, the plumber who was suspected of the theft. (Source: The Adventure of the Blue Carbuncle — The Theft at the Cosmopolitan)

  Sources:
    - The Adventure of the Blue Carbuncle — The Theft at the Cosmopolitan
    - The Adventure of the Blue Carbuncle — Holmes Lets Him Go
    - The Red-Headed League — The Identity of the Assistant
  Latency: 2938ms

Q2: What was the Hydraulic Press used for and who was using it?
Answer: The Hydraulic Press was used to stamp counterfeit half-crowns from a base alloy of nickel and tin. It was operated by Colonel Stark, whose real name was Doctor Becher. The press was part of a scheme to produce counterfeit coins, and the cover story about using it for compressing fuller's earth was a ruse to avoid suspicion. (Source: The Adventure of the Engineer's Thumb — The Coining Operation)

  Sources:
    - The Adventure of the Engineer's Thumb — The Hydraulic Press
    - The Adventure of the Engineer's Thumb — The Hydraulic Press
    - The Adventure of the Engineer's Thumb — The Coining Operation
  Latency: 3228ms

Q3: Five years earlier, What was the King (the then Crown Prince of Bohemia) attached to and where was her house that the King got burgled twice?
Answer: Five years earlier, the King (then Crown Prince of Bohemia) was attached to a celebrated American opera contralto named Irene Adler. They had met in Warsaw. The King had her house, Briony Lodge, located at Serpentine Avenue, St John's Wood, burgled twice (source: A Scandal in Bohemia — The Photograph).

  Sources:
    - A Scandal in Bohemia — The Photograph
    - A Scandal in Bohemia — The Photograph
    - A Scandal in Bohemia — Irene Adler Wins
  Latency: 2942ms
----------------------------------------------------------------------------------------------------------------------------------------
  Run python3 mp2_rag.py validate

  Validating 2 questions from predefined_questions.jsonl…

→ Embedding 1 texts in batches of 100…
  → Batch 1: 1 embeddings
  → 1 embeddings generated
  ✓ q1_red_headed_league_assistant
      Q: Who was the assistant working at Jabez Wilson's pawnshop, and what was his real identity?
      Cited: 01_red_headed_league.txt
      Expected: 01_red_headed_league.txt
      Facts matched: 1/5
      Latency: 2733ms

→ Embedding 1 texts in batches of 100…
  → Batch 1: 1 embeddings
  → 1 embeddings generated
  ✓ q2_speckled_band_murder_weapon
      Q: What killed Julia Stoner at Stoke Moran, and how was the method discovered?
      Cited: 02_speckled_band.txt
      Expected: 02_speckled_band.txt
      Facts matched: 4/6
      Latency: 2416ms

  Source-match: 2/2

  Validating 3 questions from learner_questions.jsonl…

→ Embedding 1 texts in batches of 100…
  → Batch 1: 1 embeddings
  → 1 embeddings generated
  ✓ q3_blue_carbuncle_thief
      Q: Whom does Holmes identify as the thief of Countess' blue carbuncle and who was falsely accused ?
      Cited: 01_red_headed_league.txt, 03_blue_carbuncle.txt
      Expected: 03_blue_carbuncle.txt
      Facts matched: 2/7
      Latency: 2437ms

→ Embedding 1 texts in batches of 100…
  → Batch 1: 1 embeddings
  → 1 embeddings generated
  ✓ q4_engineers_thumb_hydraulic_press
      Q: What was the Hydraulic Press used for and who was using it?
      Cited: 04_engineers_thumb.txt
      Expected: 04_engineers_thumb.txt
      Facts matched: 2/5
      Latency: 2609ms

→ Embedding 1 texts in batches of 100…
  → Batch 1: 1 embeddings
  → 1 embeddings generated
  ✓ q5_scandal_in_bohemia_kings_attachment
      Q: Five years earlier, What was the King (the then Crown Prince of Bohemia) attached to and where was her house that the King got burgled twice?
      Cited: 05_scandal_in_bohemia.txt
      Expected: 05_scandal_in_bohemia.txt
      Facts matched: 2/6
      Latency: 2253ms  

## What Worked
Chunk size of 500 characters and overlap of 80 characters worked fine for this text. 

## What didn't work
(One specific failure mode you hit. Did chunks get too small ? Did retrieval pull the wrong section ? Did the LLM Hallucinate ?)
I tried using QDRANT_URL from .env file. url=os.getenv("QDRANT_URL"). This did not work, eventually only hardcoded value worked. 
One observation was - the validate function gave Facts matched as 0 (numerator) for questions from learner_questions.jsonl, this was due to expected_facts initially containing longer texts. This got fixed once I adjusted the expected_facts list with shorter texts.

## What I'd change
Instead of using OPENAI embedding and LLM model, I would try and use Ollama to see the ask and validate results. Also I would try hybrid retrieval model.

## One surprise
(Anything from the build that genuinely surprised you. Could be technical —
"text-embedding-3-small was way better than I expected on rare names" —
or pedagogical — "I didn't realize how much hybrid retrieval was helping
until I tried dense-only here.")
I am surprised to see that the text-embedding-3-small working well for this, I think thats because the corpus was having plain text files, with no complications like tables or images. 