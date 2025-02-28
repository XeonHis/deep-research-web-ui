import { streamText } from 'ai'
import { z } from 'zod'
import { parseStreamingJson, type DeepPartial } from '~/utils/json'

import { trimPrompt } from './ai/providers'
import { languagePrompt, systemPrompt } from './prompt'
import zodToJsonSchema from 'zod-to-json-schema'
import { useAiModel } from '~/composables/useAiProvider'
import type { Locale } from '~/components/LangSwitcher.vue'
import type { DeepResearchNode } from '~/components/DeepResearch/DeepResearch.vue'

export type ResearchResult = {
  learnings: ProcessedSearchResult['learnings']
}

export interface WriteFinalReportParams {
  prompt: string
  learnings: ProcessedSearchResult['learnings']
  language: string
}

// Used for streaming response
export type SearchQuery = z.infer<typeof searchQueriesTypeSchema>['queries'][0]
export type PartialSearchQuery = DeepPartial<SearchQuery>
export type ProcessedSearchResult = z.infer<typeof searchResultTypeSchema>
export type PartialProcessedSearchResult = DeepPartial<ProcessedSearchResult>

export type ResearchStep =
  | {
      type: 'generating_query'
      result: PartialSearchQuery
      nodeId: string
      parentNodeId?: string
    }
  | { type: 'generating_query_reasoning'; delta: string; nodeId: string }
  | {
      type: 'generated_query'
      query: string
      result: PartialSearchQuery
      nodeId: string
    }
  | { type: 'searching'; query: string; nodeId: string }
  | { type: 'search_complete'; results: WebSearchResult[]; nodeId: string }
  | {
      type: 'processing_serach_result'
      query: string
      result: PartialProcessedSearchResult
      nodeId: string
    }
  | {
      type: 'processing_serach_result_reasoning'
      delta: string
      nodeId: string
    }
  | {
      type: 'node_complete'
      result?: ProcessedSearchResult
      nodeId: string
    }
  | { type: 'error'; message: string; nodeId: string }
  | { type: 'complete'; learnings: ProcessedSearchResult['learnings'] }

/**
 * Schema for {@link generateSearchQueries} without dynamic descriptions
 */
export const searchQueriesTypeSchema = z.object({
  queries: z.array(
    z.object({
      query: z.string(),
      researchGoal: z.string(),
    }),
  ),
})

// take an user query, return a list of SERP queries
export function generateSearchQueries({
  query,
  numQueries = 3,
  learnings,
  language,
  searchLanguage,
}: {
  query: string
  language: string
  numQueries?: number
  // optional, if provided, the research will continue from the last learning
  learnings?: string[]
  /** Force the LLM to generate serp queries in a certain language */
  searchLanguage?: string
}) {
  const schema = z.object({
    queries: z
      .array(
        z
          .object({
            query: z.string().describe('The SERP query.'),
            researchGoal: z
              .string()
              .describe(
                'First talk about the goal of the patent search that this query is meant to accomplish, then go deeper into how to advance the patent development once the results are found, mention additional technical aspects to explore. Be as specific as possible, especially for additional technical aspects. JSON reserved words should be escaped.',
              ),
          })
          .required({ query: true, researchGoal: true }),
      )
      .describe(`List of SERP queries, max of ${numQueries}`),
  })
  const jsonSchema = JSON.stringify(zodToJsonSchema(schema))
  let lp = languagePrompt(language)

  if (searchLanguage && searchLanguage !== language) {
    lp += ` Use ${searchLanguage} for the SERP queries.`
  }
  const prompt = [
    `Given the following invention description from the user, generate a list of SERP queries to research prior art and technical details for this patent. Your queries should be DIVERGENT and CREATIVE to help identify unique aspects of the invention and avoid conflicts with existing patents. 
    
    Focus on:
    1. Novel combinations of technologies or approaches
    2. Unconventional applications or implementations
    3. Alternative technical solutions to the same problem
    4. Edge cases and boundary conditions
    5. Different industries or fields where similar technology might exist
    
    Return a maximum of ${numQueries} queries, but feel free to return less if the original description is clear. Each query should explore a different technical dimension or perspective of the invention: <invention>${query}</invention>\n\n`,
    learnings
      ? `Here are some findings from previous searches, use them to generate more specific and divergent queries that explore UNEXPLORED technical aspects and potential novel applications: ${learnings.join(
          '\n',
        )}`
      : '',
    `You MUST respond in JSON matching this JSON schema: ${jsonSchema}`,
    lp,
  ].join('\n\n')
  return streamText({
    model: useAiModel(),
    system: systemPrompt(),
    prompt,
    onError({ error }) {
      throwAiError('generateSearchQueries', error)
    },
  })
}

export const searchResultTypeSchema = z.object({
  learnings: z.array(
    z.object({
      url: z.string(),
      learning: z.string(),
      /** This is added in {@link deepResearch} */
      title: z.string().optional(),
    }),
  ),
  followUpQuestions: z.array(z.string()),
})

function processSearchResult({
  query,
  results,
  numLearnings = 5,
  numFollowUpQuestions = 3,
  language,
}: {
  query: string
  results: WebSearchResult[]
  language: string
  numLearnings?: number
  numFollowUpQuestions?: number
}) {
  const schema = z.object({
    learnings: z
      .array(
        z.object({
          url: z
            .string()
            .describe('The source URL from which this patent information was extracted'),
          learning: z
            .string()
            .describe(
              'A detailed, information-dense technical insight relevant to the patent. Include specific technical details, implementations, materials, methods, processes, and any metrics or measurements when available',
            ),
        }),
      )
      .describe(
        `Collection of key technical insights for patent development extracted from search results, each with its source URL. Maximum of ${numLearnings} insights.`,
      ),
    followUpQuestions: z
      .array(z.string())
      .describe(
        `List of relevant follow-up questions to explore technical aspects of the invention further, designed to uncover additional patentable features or prior art. Maximum of ${numFollowUpQuestions} questions.`,
      ),
  })
  const jsonSchema = JSON.stringify(zodToJsonSchema(schema))
  const contents = results.map((item) => trimPrompt(item.content))
  const prompt = [
    `Given the following contents from a SERP search for the query <query>${query}</query>, extract key technical insights relevant for patent development. 
    
    Focus on DIVERGENT and CREATIVE aspects that could make the patent unique:
    1. Identify technical gaps or limitations in existing solutions that your invention could address
    2. Extract insights about unconventional applications or implementations
    3. Note any novel combinations of technologies or approaches
    4. Identify technical aspects that differentiate from existing patents
    5. Look for cross-industry applications or unexpected use cases
    
    For each insight, include the source URL. Return a maximum of ${numLearnings} insights, but feel free to return less if the contents are clear. Make sure each insight is unique and focuses on different technical aspects. The insights should be as detailed and technically dense as possible. Include specific technical implementations, materials, methods, processes, and any exact metrics, measurements, or specifications. Also generate up to ${numFollowUpQuestions} follow-up questions that could help explore additional patentable features or identify potential prior art.`,
    `<contents>${contents
      .map(
        (content, index) =>
          `<content url="${results[index].url}">\n${content}\n</content>`,
      )
      .join('\n')}</contents>`,
    `You MUST respond in JSON matching this JSON schema: ${jsonSchema}`,
    languagePrompt(language),
  ].join('\n\n')

  return streamText({
    model: useAiModel(),
    system: systemPrompt(),
    prompt,
    onError({ error }) {
      throwAiError('processSearchResult', error)
    },
  })
}

export function writeFinalReport({
  prompt,
  learnings,
  language,
}: WriteFinalReportParams) {
  const learningsString = trimPrompt(
    learnings
      .map(
        (learning) =>
          `<learning url="${learning.url}">\n${learning.learning}\n</learning>`,
      )
      .join('\n'),
  )
  const _prompt = [
    `Given the following invention description from the user, write a comprehensive patent document using the technical insights gathered from research. 
    
    Focus on HIGHLIGHTING THE UNIQUENESS and NOVELTY of the invention:
    1. Clearly differentiate from existing patents and prior art
    2. Emphasize novel combinations of technologies or approaches
    3. Describe unconventional applications or implementations
    4. Articulate technical advantages over existing solutions
    5. Include broad claims that protect the core innovation while being specific enough to be defensible
    
    Structure the document with proper patent sections including:
    - Title (concise, broad, descriptive)
    - Abstract (brief summary of the invention)
    - Background (problem being solved, limitations of existing solutions)
    - Summary (overview of the invention and its advantages)
    - Detailed Description (comprehensive technical details with examples)
    - Claims (hierarchical structure with independent and dependent claims)
    
    Make it as technically detailed as possible, include ALL the key technical insights from research.`,
    `<invention>${prompt}</invention>`,
    `Here are all the technical insights from previous research:`,
    `<insights>\n${learningsString}\n</insights>`,
    `Write the patent document using Markdown. When citing information, use numbered citations with superscript numbers in square brackets (e.g., [1], [2], [3]). Each citation should correspond to the index of the source in your insights list. DO NOT include the actual URLs in the document text - only use the citation numbers.`,
    languagePrompt(language),
    `## Patent Document`,
  ].join('\n\n')

  return streamText({
    model: useAiModel(),
    system: systemPrompt(),
    prompt: _prompt,
    onError({ error }) {
      throwAiError('writeFinalReport', error)
    },
  })
}

function childNodeId(parentNodeId: string, currentIndex: number) {
  return `${parentNodeId}-${currentIndex}`
}

export async function deepResearch({
  query,
  breadth,
  maxDepth,
  languageCode,
  searchLanguageCode,
  learnings,
  onProgress,
  currentDepth,
  nodeId = '0',
  retryNode,
}: {
  query: string
  breadth: number
  maxDepth: number
  /** The language of generated response */
  languageCode: Locale
  /** The language of SERP query */
  searchLanguageCode?: Locale
  /** Accumulated learnings from all nodes visited so far */
  learnings?: Array<{ url: string; learning: string }>
  currentDepth: number
  /** Current node ID. Used for recursive calls */
  nodeId?: string
  /** The Node ID to retry. Passed from DeepResearch.vue */
  retryNode?: DeepResearchNode
  onProgress: (step: ResearchStep) => void
}): Promise<ResearchResult> {
  const { t } = useNuxtApp().$i18n
  const language = t('language', {}, { locale: languageCode })
  const searchLanguage = searchLanguageCode
    ? t('language', {}, { locale: searchLanguageCode })
    : undefined
  const globalLimit = usePLimit()

  try {
    let searchQueries: Array<PartialSearchQuery & { nodeId: string }> = []

    // If retryNode is provided and not a root node, just use the query from the node
    if (retryNode && retryNode.id !== '0') {
      nodeId = retryNode.id
      searchQueries = [
        {
          query: retryNode.label,
          researchGoal: retryNode.researchGoal,
          nodeId,
        },
      ]
    }
    // Otherwise (fresh start or retrying on root node)
    else {
      const searchQueriesResult = generateSearchQueries({
        query,
        learnings: learnings?.map((item) => item.learning),
        numQueries: breadth,
        language,
        searchLanguage,
      })

      for await (const chunk of parseStreamingJson(
        searchQueriesResult.fullStream,
        searchQueriesTypeSchema,
        (value) => !!value.queries?.length && !!value.queries[0]?.query,
      )) {
        if (chunk.type === 'object' && chunk.value.queries) {
          // Temporary fix: Exclude queries that equals `undefined`
          // Currently only being reported to be seen on GPT-4o, where the model simply returns `undefined` for certain questions
          // https://github.com/AnotiaWang/deep-research-web-ui/issues/7
          searchQueries = chunk.value.queries
            .filter((q) => q.query !== 'undefined')
            .map((q, i) => ({
              ...q,
              nodeId: childNodeId(nodeId, i),
            }))
          for (let i = 0; i < searchQueries.length; i++) {
            onProgress({
              type: 'generating_query',
              result: searchQueries[i],
              nodeId: searchQueries[i].nodeId,
              parentNodeId: nodeId,
            })
          }
        } else if (chunk.type === 'reasoning') {
          // Reasoning part goes to the parent node
          onProgress({
            type: 'generating_query_reasoning',
            delta: chunk.delta,
            nodeId,
          })
        } else if (chunk.type === 'error') {
          onProgress({
            type: 'error',
            message: chunk.message,
            nodeId,
          })
          break
        } else if (chunk.type === 'bad-end') {
          onProgress({
            type: 'error',
            message: t('invalidStructuredOutput'),
            nodeId,
          })
          break
        }
      }

      onProgress({
        type: 'node_complete',
        nodeId,
      })

      for (const searchQuery of searchQueries) {
        onProgress({
          type: 'generated_query',
          query: searchQuery.query!,
          result: searchQuery,
          nodeId: searchQuery.nodeId,
        })
      }
    }

    // Run in parallel and limit the concurrency
    const results = await Promise.all(
      searchQueries.map((searchQuery) =>
        globalLimit(async () => {
          if (!searchQuery?.query) {
            return {
              learnings: [],
              visitedUrls: [],
            }
          }
          onProgress({
            type: 'searching',
            query: searchQuery.query,
            nodeId: searchQuery.nodeId,
          })
          try {
            // search the web
            const results = await useWebSearch()(searchQuery.query, {
              maxResults: 5,
              lang: languageCode,
            })
            console.log(
              `[DeepResearch] Searched "${searchQuery.query}", found ${results.length} contents`,
            )

            onProgress({
              type: 'search_complete',
              results,
              nodeId: searchQuery.nodeId,
            })
            // Breadth for the next search is half of the current breadth
            const nextBreadth = Math.ceil(breadth / 2)

            const searchResultGenerator = processSearchResult({
              query: searchQuery.query,
              results,
              numFollowUpQuestions: nextBreadth,
              language,
            })
            let searchResult: PartialProcessedSearchResult = {}

            for await (const chunk of parseStreamingJson(
              searchResultGenerator.fullStream,
              searchResultTypeSchema,
              (value) => !!value.learnings?.length,
            )) {
              if (chunk.type === 'object') {
                searchResult = chunk.value
                onProgress({
                  type: 'processing_serach_result',
                  result: chunk.value,
                  query: searchQuery.query,
                  nodeId: searchQuery.nodeId,
                })
              } else if (chunk.type === 'reasoning') {
                onProgress({
                  type: 'processing_serach_result_reasoning',
                  delta: chunk.delta,
                  nodeId: searchQuery.nodeId,
                })
              } else if (chunk.type === 'error') {
                onProgress({
                  type: 'error',
                  message: chunk.message,
                  nodeId: searchQuery.nodeId,
                })
                break
              } else if (chunk.type === 'bad-end') {
                onProgress({
                  type: 'error',
                  message: t('invalidStructuredOutput'),
                  nodeId: searchQuery.nodeId,
                })
                break
              }
            }
            console.log(
              `Processed search result for ${searchQuery.query}`,
              searchResult,
            )
            // Assign URL titles to learnings
            searchResult.learnings = searchResult.learnings?.map((learning) => {
              return {
                ...learning,
                title: results.find((r) => r.url === learning.url)?.title,
              }
            })
            const allLearnings = [
              ...(learnings ?? []),
              ...(searchResult.learnings ?? []),
            ]
            const nextDepth = currentDepth + 1

            onProgress({
              type: 'node_complete',
              result: {
                learnings: searchResult.learnings ?? [],
                followUpQuestions: searchResult.followUpQuestions ?? [],
              },
              nodeId: searchQuery.nodeId,
            })

            if (
              nextDepth <= maxDepth &&
              searchResult.followUpQuestions?.length
            ) {
              console.warn(
                `Researching deeper, breadth: ${nextBreadth}, depth: ${nextDepth}`,
              )

              const nextQuery = `
              Previous research goal: ${searchQuery.researchGoal}
              Follow-up research directions: ${searchResult.followUpQuestions
                .map((q) => `\n${q}`)
                .join('')}
            `.trim()

              // Add concurrency by 1, and do next recursive search
              globalLimit.concurrency++
              try {
                const r = await deepResearch({
                  query: nextQuery,
                  breadth: nextBreadth,
                  maxDepth,
                  learnings: allLearnings,
                  onProgress,
                  currentDepth: nextDepth,
                  nodeId: searchQuery.nodeId,
                  languageCode,
                })
                return r
              } catch (error) {
                throw error
              } finally {
                globalLimit.concurrency--
              }
            } else {
              return {
                learnings: allLearnings,
              }
            }
          } catch (e: any) {
            console.error(
              `Error in node ${searchQuery.nodeId} for query ${searchQuery.query}`,
              e,
            )
            onProgress({
              type: 'error',
              message: e.message,
              nodeId: searchQuery.nodeId,
            })
            return {
              learnings: [],
            }
          }
        }),
      ),
    )
    // Conclude results
    // Deduplicate
    const urlMap = new Map<string, true>()
    const finalLearnings: ProcessedSearchResult['learnings'] = []

    for (const result of results) {
      for (const learning of result.learnings) {
        if (!urlMap.has(learning.url)) {
          urlMap.set(learning.url, true)
          finalLearnings.push(learning)
        }
      }
    }
    // Complete should only be called once
    if (nodeId === '0') {
      onProgress({
        type: 'complete',
        learnings: finalLearnings,
      })
    }
    return {
      learnings: finalLearnings,
    }
  } catch (error: any) {
    console.error(error)
    onProgress({
      type: 'error',
      message: error?.message ?? 'Something went wrong',
      nodeId,
    })
    return {
      learnings: [],
    }
  }
}
