"""
Prompt templates for agentic chunking strategy.

This module contains all the prompt templates used by the agentic chunker
to guide LLM decision-making during the chunking process.
"""

from langchain_core.prompts import ChatPromptTemplate


# Prompt for extracting propositions from text (fallback if hub.pull fails)
PROPOSITION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at breaking down complex text into atomic, self-contained propositions.

Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.

1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible.

2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition.

3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to.

4. Present the results as a list of strings, formatted in JSON.

Example:
Input: "Title: Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content:
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare's scratch or form and a lapwing's nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."

Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter", "Hares
were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
that hares laid eggs.", "A hare's scratch or form and a lapwing's nest look very similar.", "Both
hares and lapwing's nests occur on grassland and are first seen in the spring.", "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America."]

Decompose the following:"""),
    ("user", "{text}")
])


# Prompt for finding a relevant chunk for a proposition
FIND_RELEVANT_CHUNK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Determine whether or not the "Proposition" should belong to any of the existing chunks.

A proposition should belong to a chunk if their meaning, direction, or intention are similar.
The goal is to group similar propositions and chunks.

If you think a proposition should be joined with a chunk, return the chunk id.
If you do not think an item should be joined with an existing chunk, just return "No chunks"

Example:
Input:
    - Proposition: "Greg really likes hamburgers"
    - Current Chunks:
        - Chunk ID: 2n4l3d
        - Chunk Name: Places in San Francisco
        - Chunk Summary: Overview of the things to do with San Francisco Places

        - Chunk ID: 93833k
        - Chunk Name: Food Greg likes
        - Chunk Summary: Lists of the food and dishes that Greg likes
Output: 93833k"""),
    ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
    ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}")
])


# Prompt for generating a new chunk summary
NEW_CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only respond with the new chunk summary, nothing else."""),
    ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}")
])


# Prompt for generating a new chunk title
NEW_CHUNK_TITLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

A good chunk title is brief but encompasses what the chunk is about.

You will be given a summary of a chunk which needs a title.

Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else."""),
    ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}")
])


# Prompt for updating an existing chunk summary
UPDATE_CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

You will be given a group of propositions which are in the chunk and the chunks current summary.

Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only respond with the chunk new summary, nothing else."""),
    ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}")
])


# Prompt for updating an existing chunk title
UPDATE_CHUNK_TITLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

A good title will say what the chunk is about.

You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else."""),
    ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}")
])

# Multi-Query Expansion (MQE) Prompt
MQE_PROMPT = """You are an expert AI assistant specializing in query expansion for a Retrieval-Augmented Generation (RAG) system. Your task is to take a user's query and generate 3 alternative versions of it. These versions should be rephrased to be more specific, to explore different sub-topics within the query, or to use different terminology, all with the goal of improving the retrieval of relevant documents.

Analyze the user's original query provided in the <query> tag. Then, generate 3 diverse and high-quality alternative queries.

**CRITICAL INSTRUCTIONS:**
1.  Output ONLY the 3 alternative queries.
2.  Each query should be on a new line.
3.  DO NOT number the queries or add any other text, preambles, or explanations.

<query>
{user_query}
</query>"""


# RAG System Prompt - Used by LLM generators for question answering
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

Instructions:
1. Use ONLY the information provided in the context to answer the question
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite the sources by mentioning the document titles when possible
4. Be concise but comprehensive in your response
5. If conflicting information exists in the context, acknowledge it

Context:
{context}

Question: {question}

Please provide a helpful and accurate answer based on the context above."""


# Conversational RAG System Prompt - Used by RAG pipeline for conversational interactions
CONVERSATIONAL_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided context documents and conversation history.

Instructions:
1. Use the provided context documents to answer questions when available
2. Reference previous conversation history when relevant
3. If no context is provided, use your general knowledge but mention this
4. Be conversational and remember what was discussed earlier
5. If you don't know something, admit it rather than guessing
6. Cite sources when using context documents
7. Ask clarifying questions when needed

The conversation history and context will help you provide more relevant and personalized responses."""
