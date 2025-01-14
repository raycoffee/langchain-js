import dotenv from 'dotenv';
import { ChatAnthropic } from '@langchain/anthropic';


dotenv.config();

const model = new ChatAnthropic({
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    model: "claude-3-haiku-20240307",
    maxTokens: 1000,
    verbose: true
});


const response = await model.invoke("write 2 lines about kolkata");

console.log(response)