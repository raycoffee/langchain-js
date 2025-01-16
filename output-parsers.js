import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser, CommaSeparatedListOutputParser } from "@langchain/core/output_parsers";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";

import dotenv from 'dotenv';

dotenv.config();

const model = new ChatAnthropic({
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    model: "claude-3-haiku-20240307",
    maxTokens: 1000,
    temperature: 0.7
})

async function callStringOutputParser(input) {
    // Create Prompt Templates

    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Generate a prompt based on a word provided by the user"], ["human", "{input}"]
    ])

    const parser = new StringOutputParser()

    // Create chain
    const chain = prompt.pipe(model).pipe(parser)


    // Call chain
    return await chain.invoke({
        input
    })


}

async function callListOutputParser(input) {

    const prompt = ChatPromptTemplate.fromTemplate("Provide 5 synonyms, separated by commas, for the follwing word: {word}")

    const outputParser = new CommaSeparatedListOutputParser()

    const chain = prompt.pipe(model).pipe(outputParser)

    return await chain.invoke({
        word: input
    })

}

async function callStructuredParser() {
    const prompt = ChatPromptTemplate.fromTemplate("Extract information from the following phrase. Formatting Instructions: {format_instructions} Phrase: {phrase}")

    const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
        name: "The name of the person",
        age: "The age of the person"
    })

    const chain = prompt.pipe(model).pipe(outputParser)

    return await chain.invoke({
        phrase: "Max is 30 years old",
        format_instructions: outputParser.getFormatInstructions()
    })
}

async function callZodOutputParser() {
    const prompt = ChatPromptTemplate.fromTemplate("Extract information from the following phrase. Formatting Instructions: {format_instructions} Phrase: {phrase}")

    const outputParser = StructuredOutputParser.fromZodSchema(
        z.object({
            recipe: z.string().describe("name of recipe"),
            ingredients: z.array(z.string()).describe("ingredients")
        })
    )

    const chain = prompt.pipe(model).pipe(outputParser)

    return await chain.invoke({
        phrase: "The ingredients for chicken keema is butter, oil, masala, corn, potato and cheese",
        format_instructions: outputParser.getFormatInstructions()
    })
}

// const response = await callStringOutputParser("shark")
// const response = await callListOutputParser("happy")
// const response = await callStructuredParser()
const response = await callZodOutputParser()

console.log(response)