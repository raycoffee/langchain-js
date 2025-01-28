import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

import { configDotenv } from "dotenv";

configDotenv()



// Load data and create Vector Store
const createVectorStore = async () => {
    const loader = new CheerioWebBaseLoader("https://python.langchain.com/docs/concepts/lcel/")

    const embeddings = new HuggingFaceInferenceEmbeddings({
        apiKey: process.env.HUGGINGFACEHUB_API_KEY,
        model: "sentence-transformers/all-MiniLM-L6-v2",
    });


    const docs = await loader.load()

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20
    })

    const splitDocs = await splitter.splitDocuments(docs)

    const vectorStore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        embeddings
    )

    return vectorStore
}

// Create retrieval chain
const createChain = async (vectorStore) => {

    const model = new ChatAnthropic({
        anthropicApiKey: process.env.ANTHROPIC_API_KEY,
        model: "claude-3-haiku-20240307",
        maxTokens: 1000,
        temperature: 0.7
    })

    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Answer the user's question based on the following context: {context}."],
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"]
    ])

    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt
    })


    const retriever = vectorStore.asRetriever({
        k: 2
    })

    const retrieverPrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
        ["user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation."]
    ])

    const historyAwareRetriever = await createHistoryAwareRetriever({
        llm: model,
        retriever,
        rephrasePrompt: retrieverPrompt
    })

    const conversationChain = await createRetrievalChain({
        combineDocsChain: chain,
        retriever: historyAwareRetriever
    })

    return conversationChain
}

const vectorStore = await createVectorStore()
const chain = await createChain(vectorStore)


// Chat history
const chatHistory = [
    new HumanMessage("Hello"),
    new AIMessage("Hi, how can I help you?"),
    new HumanMessage("My name is Ray"),
    new AIMessage("Hi Leon, how can I help you?"),
    new HumanMessage("What is a RunnableSequence?"),
    new AIMessage(`A RunnableSequence is a composition primitive in the context of concurrent programming that allows you to "chain" multiple runnables (executable units of code) sequentially.`)
]

const response = await chain.invoke({
    input: "What is it?",
    chat_history: chatHistory
})

console.log(response.answer, "😊")