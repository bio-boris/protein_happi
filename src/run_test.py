from src.search import search_impl
from src.models import SearchRequest


async def single_test() -> None:
    # A Quick Test
    data = {
        "query_sequences": [
            {
                "id": "string",
                "sequence": "MPETTNLPAGPAGPDRPDSPDSPDSPDSPDRRLKGRGIPDAPGNRFERLHVEIDVGAMAEMQTVDPEWEAAPPRTVFYRDETQSIVSTNASPDLNFDASLNPYRGCEHGCSYCYARPYHEYLGFNSGIDFETRILVKENAGALLEKELGSKKWKPKTLVCSGVTDPYQPVEKKLKITRRCLEVLEHFRNPVGIITKNHLVTRDIDHLGVLAREHSAACVYISITTLDRNLAKVLEPRASSPSFRLRAVKELSEAGIPVGVSLGPTIPGLNDHEMPAILEAASDHGARTAFYILLRLPHGVSKMFSDWLGCHFPQRKEKVLGRLRELRGGKLNDSRFGVRFKGEGPLASEIESLFRVSARKCGLHRAMPELSCAAFRRAAGGGQMELF",
            }
        ],
        "similarity_threshold": 0,
        "best_hit_only": False,
        "max_hits": 5,
        "return_query_embeddings": False,
        "return_hit_embeddings": False,
    }

    query = SearchRequest(**data)

    response = await search_impl(query)

    print(response)


async def genome_test() -> None:
    import json
    from protein_search_evals.utils import read_fasta

    fasta_file = "/scratch/abrace/data/ecoli/UP000000625_83333.fasta"
    results_file = "/scratch/abrace/data/ecoli/exact_vs_ivf/UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe256-topk100.json"

    sequences = read_fasta(fasta_file)
    query_sequences = [{"id": seq.tag, "sequence": seq.sequence} for seq in sequences]

    data = {
        "query_sequences": query_sequences,
        "similarity_threshold": 0,
        "best_hit_only": False,
        "max_hits": 100,
        "return_query_embeddings": False,
        "return_hit_embeddings": False,
    }

    query = SearchRequest(**data)

    response = await search_impl(query)

    with open(results_file, "w") as fp:
        json.dump(response.model_dump(), fp, indent=2)

if __name__ == "__main__":
    import asyncio

    asyncio.run(single_test())
    asyncio.run(single_test())
    # asyncio.run(genome_test())
    # asyncio.run(genome_test())
