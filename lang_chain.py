from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.globals import set_debug
from detalhes_imagem_modelo import DetalhesImagemModelo

set_debug(True)

llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model=GEMINI_FLASH,
)

imagem = encode_image("dados\exemplo_grafico.jpg")

template_analisador = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            Assuma que você é um analisador de imagens. A sua tarefa principal consiste em: analisar uma imagem e extrair informações importantes de forma objetiva.
            
            # FORMATO DE SAÍDA
            Descrição da Imagem: 'Coloque a sua descrição da imagem aqui'
            Rótulos: 'Coloque uma lista com três termos chave separados por vírgula'

            """
        ),
        (
            "user", 
            [
                {
                    "type": "text",
                    "text": "Descreva a imagem: "
                },
                {
                    "type": "image_url",
                    "image_url": {"url":"data:image/jpeg;base64,{imagem}"}
                }
            ]
        )
    ]
)

cadeia_analise_imagem = template_analisador | llm | StrOutputParser()

parser_json_imagem = JsonOutputParser(
    pydantic_object=DetalhesImagemModelo
)

template_resposta = PromptTemplate(
    template="""
    Gere um resumo, utilizando uma linguagem clara e objetiva, focada no público brasileiro. Adeia é que a comunicação do resultado seja o mais fácil possível, priorizando registros para consultas posteriores.
    
    # O Resultado da imagem
    {resposta_cadeia_alalise_imagem}

    # FORMATO DE SAIDA
    {formato_saida}
    """,
    input_variables=["resposta_cadeia_alalise_imagem"],
    partial_variables={"formato_saida": parser_json_imagem.get_format_instructions()}
)

# llm_maritaca = ChatMaritalk(
#     api_key=MARITACA_API_KEY,
#     model=MARITACA_SABIA,
# )

cadeia_resumo = template_resposta | llm | parser_json_imagem

cadeia_completa = (cadeia_analise_imagem | cadeia_resumo)

resposta = cadeia_completa.invoke({"imagem":imagem})

print(resposta)