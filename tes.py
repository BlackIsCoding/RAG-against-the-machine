
from t import Indexer


# ---------------------------------------------------------
# The same chunks you use in your test
# ---------------------------------------------------------

chunks = [
    """
    The scheduler is responsible for deciding which sequence groups should
    execute during each scheduling iteration. It considers the number of
    tokens requested by each sequence, the available KV cache blocks, and
    the configured token budget. Waiting requests may remain in the queue
    when there is not enough memory or token capacity to schedule them.
    Running requests can continue generating tokens as long as the scheduler
    can allocate the required KV cache resources.
    """,

    """
    The KV cache stores the key and value tensors produced by the attention
    layers during model execution. Instead of allocating one large contiguous
    memory region for every request, the system divides the cache into blocks.
    These blocks can be allocated dynamically to sequences and released when
    sequences finish. Block-based allocation makes it possible to handle
    requests with different sequence lengths without requiring every request
    to reserve its maximum possible cache size.
    """,

    """
    Continuous batching allows new requests to join execution while other
    requests are still generating tokens. The scheduler does not need to wait
    until an entire batch has completed before considering new work. During
    every scheduling iteration, it can combine currently running sequences
    with newly admitted sequences, subject to the token budget and available
    KV cache capacity. This improves hardware utilization when requests have
    different generation lengths.
    """,

    """
    PagedAttention is an attention mechanism designed around non-contiguous
    KV cache storage. The KV cache is divided into fixed-size blocks, while
    logical sequences can reference a collection of physical blocks. This
    design avoids the requirement that all tokens belonging to one sequence
    occupy one contiguous memory region. The block mapping allows memory to
    be allocated and released more flexibly as sequences grow.
    """,

    """
    When a request enters the engine, it is first represented by an internal
    request object containing information such as its prompt, sampling
    parameters, identifiers, and generation state. The request is eventually
    converted into the internal sequence representation used by scheduling.
    The scheduler then determines when the request can enter execution and
    how many tokens should be processed during a particular iteration.
    """,

    """
    A sequence represents the state of an individual generated sequence.
    It keeps track of information such as the token IDs, the number of tokens
    that have already been computed, its logical status, and its relationship
    with the KV cache. Sequence groups can contain one or more sequences that
    belong to the same request or decoding operation. Scheduling decisions
    operate on these internal representations rather than directly on raw
    API requests.
    """,

    """
    During execution, the worker is responsible for performing computation
    on the available accelerator. The scheduler determines which sequences
    should execute, while the worker receives the resulting model inputs and
    performs the forward computation. This separation allows scheduling
    decisions to remain independent from the low-level implementation used
    to execute the neural network.
    """,

    """
    The model runner prepares the tensors required by the model after the
    engine has determined which requests should execute. It constructs input
    IDs, positions, attention-related information, and other execution
    metadata. The runner then invokes the model and returns the resulting
    hidden states or logits required for the next generation step.
    """,

    """
    Prefix caching can reuse KV cache blocks corresponding to a prefix that
    has already been processed. When multiple requests share identical
    prefixes, previously computed key and value tensors can potentially be
    reused instead of recomputing the same attention states. The cache
    manager must track which blocks correspond to which cached prefixes and
    determine whether a requested prefix can be reused safely.
    """,

    """
    The engine maintains different states for requests as they progress
    through inference. A request can initially wait for scheduling, become
    running when resources are available, and eventually finish when the
    model produces a stop condition or another termination condition occurs.
    State transitions are important because the scheduler treats waiting and
    running requests differently when constructing the next execution batch.
    """,

    """
    Memory pressure can prevent the scheduler from admitting additional
    sequences even when those sequences are otherwise valid. A request may
    require more KV cache blocks than are currently available. In this case,
    the scheduler must either postpone the request, preempt another sequence,
    or otherwise manage the available resources depending on the configured
    scheduling policy. KV cache availability therefore directly affects
    scheduling decisions.
    """,

    """
    Token budgeting controls how much computation the scheduler is allowed
    to schedule during one iteration. The scheduler must balance the number
    of tokens assigned to existing decoding requests with tokens required
    for requests that are still in the prompt processing phase. Increasing
    the token budget can allow more work in one iteration, but memory
    availability and other scheduling constraints still limit execution.
    """,

    """
    During autoregressive generation, a sequence repeatedly enters the
    scheduling and execution process to produce additional tokens. The
    number of already computed tokens is tracked so that the engine knows
    which part of the sequence still requires computation. This state is
    especially important when requests are processed incrementally through
    multiple scheduling iterations.
    """,

    """
    The request management layer and the model execution layer have
    different responsibilities. Request management handles admission,
    scheduling, request state, cancellation, and resource decisions.
    Model execution handles tensor preparation and neural network computation.
    Keeping these responsibilities separate makes it possible to change
    scheduling policies without rewriting the underlying model execution
    implementation.
    """,

    """
    KV cache blocks are allocated as sequences require additional capacity.
    When a sequence finishes, the blocks associated with it can be returned
    to the pool so that future requests can use the released memory. The
    block manager therefore acts as an important connection between memory
    management and scheduling because the scheduler must know whether enough
    blocks are available before admitting more work.
    """,

    """
    A decoding request generally processes a small number of new tokens at
    each iteration, while a newly arriving request may initially require
    processing a much larger prompt. The scheduler has to account for these
    different workloads when constructing a batch. This distinction between
    prompt processing and token-by-token decoding is one reason why a
    continuous batching scheduler needs explicit token budgeting.
    """,

    """
    The scheduler does not simply select requests based on arrival order.
    It must consider scheduling policy, resource availability, token limits,
    sequence state, and KV cache capacity. Consequently, two requests that
    arrive at nearly the same time may not necessarily be executed together.
    The final batch is the result of applying these constraints to the
    current state of the engine.
    """,

    """
    Attention computation depends on the key and value states associated
    with previously processed tokens. Keeping these states in the KV cache
    prevents the system from recomputing the entire history for every newly
    generated token. Efficient KV cache management is therefore essential
    for high-throughput autoregressive inference, especially when many
    sequences are being generated concurrently.
    """,

    """
    When a sequence is finished because it generated a stop token, reached
    a configured maximum length, or encountered another termination
    condition, it no longer needs to participate in future scheduling
    iterations. Its resources can subsequently be released. Removing
    completed sequences allows their KV cache blocks and other resources
    to become available to waiting requests.
    """,

    """
    The overall inference loop can be viewed as a repeated cycle of request
    management, scheduling, input preparation, model execution, and output
    processing. The scheduler selects work, the model runner prepares and
    executes the corresponding tensors, and the resulting token information
    updates the internal sequence state. The cycle repeats until individual
    requests reach a terminal state.
    """,

    """
    Efficient inference requires coordination between computation and memory.
    A scheduler that admits too many sequences may exhaust the KV cache,
    while a scheduler that is too conservative may leave accelerator
    resources underutilized. Modern inference engines therefore treat token
    capacity, memory blocks, batching decisions, and sequence progress as
    interconnected parts of the scheduling problem.
    """
]


# ---------------------------------------------------------
# Questions + ground truth
#
# Each number represents a chunk ID that contains information
# needed to answer the question.
# ---------------------------------------------------------

tests = [
    {
        "question": "Why would a request remain waiting even though it is valid?",
        "expected": {0, 10, 14, 20},
    },
    {
        "question": "How does the system decide how much work to execute in one iteration?",
        "expected": {0, 11, 15, 20},
    },
    {
        "question": "Why is memory allocation connected to scheduling decisions?",
        "expected": {0, 10, 14, 20},
    },
    {
        "question": "How can multiple users with different generation lengths be processed together?",
        "expected": {2, 15, 20},
    },
    {
        "question": "What happens to the memory used by a sequence after generation finishes?",
        "expected": {1, 14, 18},
    },
    {
        "question": "How does the engine avoid recalculating information from previous tokens?",
        "expected": {8, 17},
    },
    {
        "question": "What is the difference between deciding what runs and actually running the neural network?",
        "expected": {6, 7, 13},
    },
    {
        "question": "Why is contiguous memory allocation inefficient for variable length requests?",
        "expected": {1, 3},
    },
    {
        "question": "How can two requests benefit from having the same beginning?",
        "expected": {8},
    },
    {
        "question": "What information does the engine need to remember between generation steps?",
        "expected": {5, 12, 17},
    },
    {
        "question": "Why can a newly arriving request be processed before another request has finished?",
        "expected": {2},
    },
    {
        "question": "How does the system balance prompt processing against token generation?",
        "expected": {11, 15},
    },
    {
        "question": "What causes a sequence to stop participating in future iterations?",
        "expected": {18},
    },
    {
        "question": "Explain the complete path from an incoming request to generated tokens.",
        "expected": {4, 0, 7, 12, 19},
    },
    {
        "question": "Which components are responsible for memory, scheduling, and model computation?",
        "expected": {0, 1, 6, 7, 14},
    },
]


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(indexer, top_k=5):
    passed = 0

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for number, test in enumerate(tests, 1):
        query = test["question"]
        expected = test["expected"]

        results = indexer.fusion(query, top_k)

        retrieved_ids = {
            chunk_id
            for chunk_id, score in results
        }

        found = retrieved_ids & expected

        # PASS if at least one relevant chunk was retrieved.
        success = len(found) > 0

        if success:
            passed += 1

        status = "PASS" if success else "FAIL"

        print(f"\n[{status}] Test {number}")
        print(f"Question: {query}")
        print(f"Expected relevant chunks: {sorted(expected)}")
        print(f"Retrieved chunks:          {sorted(retrieved_ids)}")
        print(f"Relevant chunks found:    {sorted(found)}")

    total = len(tests)
    percentage = (passed / total) * 100

    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{total} passed ({percentage:.1f}%)")
    print("=" * 70)


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

indexer = Indexer(chunks)

# Build/reuse the BM25 index
indexer.bm25s_indexing()

evaluate(indexer, top_k=5)
