import 'dotenv/config'
import { ChatOpenAI } from "@langchain/openai";


const model = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY
})