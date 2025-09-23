from src.search import logger, SearchRequest, search_impl


async def single_test() -> None:
    logger.info("Starting single test")
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
    logger.info(f"Single test completed. Found {len(response.hits)} result groups")


async def genome_test() -> None:
    import json
    from protein_search_evals.utils import read_fasta

    logger.info("Starting genome test")

    fasta_file = "/scratch/abrace/data/ecoli/UP000000625_83333.fasta"
    results_file = "/scratch/abrace/data/ecoli/exact_vs_ivf/UP000000625_83333-search-results-trembl-esm3b-faesm-bs128-ubinary-ivf-nprobe256-topk100.json"

    logger.info(f"Reading FASTA file: {fasta_file}")
    sequences = read_fasta(fasta_file)
    query_sequences = [{"id": seq.tag, "sequence": seq.sequence} for seq in sequences]
    logger.info(f"Loaded {len(query_sequences)} sequences from FASTA file")

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

    logger.info(f"Writing results to: {results_file}")
    with open(results_file, "w") as fp:
        json.dump(response.model_dump(), fp, indent=2)
    logger.info("Genome test completed successfully")


if __name__ == "__main__":
    import asyncio
    from .logging_config import setup_logging

    # Setup logging for standalone execution
    setup_logging(log_level="INFO", service_name="search_test")

    logger.info("Starting search tests")
    asyncio.run(single_test())
    asyncio.run(single_test())
