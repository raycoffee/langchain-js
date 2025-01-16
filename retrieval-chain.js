import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { MemoryVectorStore } from "langchain/vectorstores/memory";


import { configDotenv } from "dotenv";

configDotenv()

const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_KEY,
    model: "sentence-transformers/all-MiniLM-L6-v2",
});


const model = new ChatAnthropic({
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    model: "claude-3-haiku-20240307",
    maxTokens: 1000,
    temperature: 0.7
})

const prompt = PromptTemplate.fromTemplate(`
    Context: {context}
    Question: {input}
    `)

// const chain = prompt.pipe(model)
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt
})

//## LOAD DATA FROM WEBPAGE
const loader = new CheerioWebBaseLoader("https://python.langchain.com/docs/concepts/lcel/")

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

//## RETRIEVE DATA
const retriever = vectorStore.asRetriever({
    k: 2
})

const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever
})


// const vector = await embeddings.embedQuery(query);



const response = await retrievalChain.invoke({
    input: "What is LCEL?",
})

// console.log(response)