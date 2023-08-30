import { ChatOpenAI } from "langchain/chat_models/openai";
import {pinecone} from "@/utils/pinecone-client";
import {PineconeStore} from "langchain/vectorstores/pinecone";
import {OpenAIEmbeddings} from "langchain/embeddings/openai";
import {ConversationalRetrievalQAChain} from "langchain/chains";
import { ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate } from "langchain/prompts";



async function initChain() {

    const chat = new ChatOpenAI({
        temperature: 0
    })
   
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX ?? '');

    /* create vectorstore*/
    const vectorStore = await PineconeStore.fromExistingIndex(
        new OpenAIEmbeddings({}),
        {
            pineconeIndex: pineconeIndex,
            textKey: 'text',
        },
    );

    return ConversationalRetrievalQAChain.fromLLM(
        chat,
        vectorStore.asRetriever(),
        {returnSourceDocuments: true}
    );

    
     // TODO set up chatbot with prompts

    // const sys_template = 'Ste senior právnik s názvom "Lawbot", ktorý je špecializovaný na vyhľadávanie odpovedí na právne otázky vo vektorovej databáze pinecone. Ak neviete odpoveď na otázku, odpovedzte: "Nepoznám odpoveď". Vždy odpovedajte v slovenskom jazyku. Ak vás niekto osloví s otázkou, ktorá nie je právne relevantná, odpovedzte: "Prosím pýtajte sa ma iba na právne relevantné otázky". Ak sa vás niekto opýta na vaše meno, odpovedzte, že sa voláte "Lawbot".'

    // return ConversationalRetrievalQAChain.fromLLM(
    //     chat,
    //     vectorStore.asRetriever(),
    //     {
    //         returnSourceDocuments: true,
    //         qaChainOptions: {type: "stuff", prompt: PromptTemplate.fromTemplate(sys_template)},
    //     }
    // );
}

export const chain = await initChain();

