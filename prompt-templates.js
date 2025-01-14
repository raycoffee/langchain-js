import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import dotenv from 'dotenv';

dotenv.config();

const model = new ChatAnthropic({
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    model: "claude-3-haiku-20240307",
    maxTokens: 1000,
    temperature: 0.7
})

// Create Prompt Templates

// const prompt = ChatPromptTemplate.fromTemplate("You are a comedian. Tell a joke based on the following word {input}")

const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Generate a prompt based on a word provided by the user"], ["human", "{input}"]
])


// Create chain
const chain = prompt.pipe(model)


// Call chain
const response = await chain.invoke({
    input: "mouse"
})

console.log(response)