import { configDotenv } from "dotenv";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";
import { RunnableSequence } from "@langchain/core/runnables";

configDotenv()

const model = new ChatAnthropic({
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    model: "claude-3-haiku-20240307",
    maxTokens: 1000,
    temperature: 0.7
})

const prompt = ChatPromptTemplate.fromTemplate(`
    You are an AI assistant.
    History: {history}
    {input}
    `)


const upstashChatHistory = new UpstashRedisChatMessageHistory({
    sessionId: "chat1",
    config: {
        url: process.env.UPSTASH_REDIS_REST_URL,
        token: process.env.UPSTASH_REDIS_REST_TOKEN
    }
})

const memory = new BufferMemory({
    memoryKey: "history",
    chatHistory: upstashChatHistory
})

// Using the Chain Classes

const chain = RunnableSequence.from([
    {
        input: (initialInput) => {
            return initialInput.input
        },
        memory: async () => {
            return await memory.loadMemoryVariables()
        }
    },

    {
        input: (previousOutput) => {
            return previousOutput.input
        },
        history: (previousOutput) => {
            return previousOutput.memory.history
        }
    },
    prompt,
    model
])

// const chain = new ConversationChain({
//     llm: model,
//     prompt,
//     memory
// })


// LCEL
// const chain = prompt.pipe(model)

// Get Responses
// console.log(await memory.loadMemoryVariables())

// const input1 = {
//     input: "I want you to know about the passphrase 'BPCL'. When asked again about the passphrase, just reply with 'BPCL'."
// }
// const response1 = await chain.invoke(input1)

// console.log(response1, "🙏")

// console.log(await memory.loadMemoryVariables())
// await memory.saveContext(input1, {
//     output: response1.content
// })

const input2 = {
    input: "What is the passphrase again?"
}
const response2 = await chain.invoke(input2)

await memory.saveContext(input2, {
    output: response2.content
})

