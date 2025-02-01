import { configDotenv } from "dotenv";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrieverTool } from "langchain/tools/retriever";
import readline from "readline";

configDotenv()

const model = new ChatAnthropic({
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    model: "claude-3-haiku-20240307",
    maxTokens: 1000,
    temperature: 0.7
})

const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_KEY,
    model: "sentence-transformers/all-MiniLM-L6-v2",
});

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

const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant called Ray."],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad")
])

// Create & Assign Tools
const searchTool = new TavilySearchResults()
const retrieverTool = createRetrieverTool(retriever, {
    name: "lcel_search",
    description: "Use this tool when searching for information about Langchain Expression Language (LCEL)"
})
const tools = [searchTool, retrieverTool]


// Create Agent
const agent = createToolCallingAgent({
    llm: model,
    tools,
    prompt,
});


// Create Agent Executor
const agentExecutor = new AgentExecutor({
    agent,
    tools
})

// Get user input
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
})

// Chat history
const chatHistory = [

]

const askQuestion = () => {


    return rl.question("User: ", async (input) => {

        if (input.toLowerCase() === 'exit') {
            rl.close()
            return
        }

        chatHistory.push(new HumanMessage(input))

        const response = await agentExecutor.invoke({
            input,
            chat_history: chatHistory
        })

        console.log("Agent: ", response.output[0].text, "😡")
        chatHistory.push(new AIMessage(response.output[0].text))
        askQuestion()
    })


}


askQuestion()