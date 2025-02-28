import { streamText } from 'ai'
import { z } from 'zod'
import { zodToJsonSchema } from 'zod-to-json-schema'

import { languagePrompt, systemPrompt } from './prompt'
import { useAiModel } from '~/composables/useAiProvider'

type PartialFeedback = DeepPartial<z.infer<typeof feedbackTypeSchema>>

export const feedbackTypeSchema = z.object({
  questions: z.array(z.string()),
})

export function generateFeedback({
  query,
  language,
  numQuestions = 3,
}: {
  query: string
  language: string
  numQuestions?: number
}) {
  const schema = z.object({
    questions: z
      .array(z.string())
      .describe(`Follow up questions to clarify the patent invention details`),
  })
  const jsonSchema = JSON.stringify(zodToJsonSchema(schema))
  const prompt = [
    `Given the following invention description from the user, ask ${numQuestions} follow up questions to clarify critical patent details. Focus on technical specifications, unique features, potential applications, and differentiators from existing solutions. Return a maximum of ${numQuestions} questions, but feel free to return less if the original description is comprehensive: <invention>${query}</invention>`,
    `You MUST respond in JSON matching this JSON schema: ${jsonSchema}`,
    languagePrompt(language),
  ].join('\n\n')

  const stream = streamText({
    model: useAiModel(),
    system: systemPrompt(),
    prompt,
    onError({ error }) {
      throwAiError('generateFeedback', error)
    },
  })

  return parseStreamingJson(
    stream.fullStream,
    feedbackTypeSchema,
    (value: PartialFeedback) => !!value.questions && value.questions.length > 0,
  )
}
