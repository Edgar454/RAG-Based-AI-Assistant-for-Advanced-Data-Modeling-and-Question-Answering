from glob import glob
import re
from tqdm.auto import tqdm

from langchain_core.documents import Document
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations , shared
from unstructured_client.models.errors import SDKError

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements



def load_and_process_document(documents_path ,
                              method= "local",
                              api_key_auth = None ,
                              server_url = None, ):
    
    """Function to load the document and preprocess them using logics and the unstructured library for layout detection
    Args
    --------------------------------------
    documents_path : path to the document to preprocess
    method : "local" or "server" , wether to process the documents localy or via the unstructured client API
    api_key_auth : key for the unstructured client API
    server_url: url of the unstructured client API

    """
    files = glob(documents_path + "\\*.pdf")


    if method == "local":
        pdf_elements = []

        #Partitioning of the pdf files based on layout
        for file in files :
            pdf_element = partition_pdf(
                                        filename=file,                  # mandatory
                                        strategy="hi_res",                                     # mandatory to use ``hi_res`` strategy
                                        extract_images_in_pdf=True,                            # mandatory to set as ``True``
                                        extract_image_block_types=["Image", "Table"],          # optional
                                        extract_image_block_to_payload=False,                  # optional
                                        extract_image_block_output_dir="Documents",  # optional - only works when ``extract_image_block_to_payload=False``
                                        )
            pdf_elements.append(pdf_element)
    
    if method == "server" and (api_key_auth is not None) and (server_url is not None) :

        # intantiating the client
        client = UnstructuredClient(
            api_key_auth= api_key_auth,
            server_url=server_url,
        )

        pdf_elements = []

        #Partitioning of the pdf files based on layout
        for file in files :
            with open(file, "rb") as f:
                data = f.read()


            req = operations.PartitionRequest(
                    partition_parameters=shared.PartitionParameters(
                        files=shared.Files(
                            content=data,
                            file_name=file,
                        ),
                        strategy=shared.Strategy.HI_RES,  
                        languages=['en'],
                    ),
                )

            try:
                resp = client.general.partition(request=req)
                pdf_element = dict_to_elements(resp.elements)
            except SDKError as e:
                print(e)

            pdf_elements.append(pdf_element)

    to_filter_out = ['APPENDIX',
                 'Acknowledgments',
                 'Bibliography',
                 'How to Contact Us',
                 'Index',
                 'Table of Contents',
                 'Revision',
                 'Audience'
                 ]

    for i, pdf_element in enumerate(pdf_elements):  # Iterate with index

        #removing the footers and the headers
        pdf_element = [elm for elm in pdf_element if elm.category not in ["Header","Footer"]]
    

        # Identify the ids of the elements to filter out
        chapter_ids = {}
        for element in tqdm(pdf_element):
            for chapter in to_filter_out:
                # Use word boundary \b to match whole words
                if re.findall(chapter.lower() + r'\b', element.text.lower()) != [] and element.category == "Title":
                    chapter_ids[element.id] = element.text

        # Build an inverse mapping from title to id
        chapter_to_id = {v: k for k, v in chapter_ids.items()}

        # Filter out the chunks with the ids corresponding to the elements to filter out
        pdf_element_filtered = [elm for elm in pdf_element if elm.metadata.parent_id not in chapter_to_id.values()]

        # Replace the original pdf_elements_s[i] with the filtered list
        pdf_elements[i] = pdf_element_filtered  # Update the original list


    # let's build the chunks by preserving element in the same title together
    chunked_elements = []
    for pdf_element in pdf_elements :
        element= chunk_by_title(pdf_element)
        chunked_elements.append(element)

    # Building the documents files for vector store
    documents = []
    for pdf_element in chunked_elements :
        for element in pdf_element:
            metadata = element.metadata.to_dict()
            del metadata["languages"]
            metadata["source"] = metadata["filename"]
            documents.append(Document(page_content=element.text, metadata=metadata))

    return documents

