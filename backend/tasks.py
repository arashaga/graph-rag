import asyncio
import traceback

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.enums import IndexingMethod
from graphrag.index.run.run_pipeline import run_pipeline
from graphrag.index.run.utils import create_callback_chain
from graphrag.callbacks.reporting import create_pipeline_reporter
from graphrag.logger.null_progress import NullProgressLogger
from graphrag.index.workflows.factory import PipelineFactory
import yaml
import os

# Simple in-memory job status store
JOB_STATUS = {}

def load_settings(settings_path: str, job_folder: str) -> GraphRagConfig:
    print(f"Loading settings from: {settings_path}")
    print(f"Job folder: {job_folder}")
    
    with open(settings_path, "r") as f:
        settings_dict = yaml.safe_load(f)
    
    print(f"Loaded settings_dict: {settings_dict.get('models', {})}")
    
    # Set both input and output dirs to the job folder
    settings_dict['input']['base_dir'] = job_folder
    settings_dict['output']['base_dir'] = job_folder
    
    # Update vector store path to inside the job folder
    if 'vector_store' in settings_dict and 'default_vector_store' in settings_dict['vector_store']:
        settings_dict['vector_store']['default_vector_store']['db_uri'] = f"{job_folder}/lancedb"

    # You can update other output-dependent dirs here if needed (like reporting, cache)
    if 'reporting' in settings_dict and 'base_dir' in settings_dict['reporting']:
        settings_dict['reporting']['base_dir'] = f"{job_folder}/logs"
    if 'cache' in settings_dict and 'base_dir' in settings_dict['cache']:
        settings_dict['cache']['base_dir'] = f"{job_folder}/cache"

    return GraphRagConfig(**settings_dict)

def run_indexing_sync(job_folder: str, method: IndexingMethod):
    """Synchronous wrapper for async build_index."""
    import asyncio

    # Use absolute path to settings.yaml in the backend directory
    settings_path = os.path.join(os.path.dirname(__file__), "settings.yaml")
    config = load_settings(settings_path, job_folder)

    async def _build():
        from graphrag.index.typing.pipeline_run_result import PipelineRunResult
        from graphrag.index.run.run_pipeline import run_pipeline
        from graphrag.index.run.utils import create_callback_chain
        from graphrag.callbacks.reporting import create_pipeline_reporter
        from graphrag.logger.null_progress import NullProgressLogger
        from graphrag.index.workflows.factory import PipelineFactory
        from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks

        logger = NullProgressLogger()
        callbacks = [create_pipeline_reporter(config.reporting, None)]
        workflow_callbacks = create_callback_chain(callbacks, logger)
        outputs = []
        pipeline = PipelineFactory.create_pipeline(config, method, False)
        workflow_callbacks.pipeline_start(pipeline.names())

        async for output in run_pipeline(
            pipeline, config, callbacks=workflow_callbacks, logger=logger, is_update_run=False,
        ):
            outputs.append(output)
        workflow_callbacks.pipeline_end(outputs)
        return outputs

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If already in an event loop, create a task and wait for it
        return asyncio.ensure_future(_build())
    else:
        return asyncio.run(_build())

def start_indexing(job_id, job_folder , method):
    """Runs the indexing, updates JOB_STATUS for UI polling."""
    try:
        print(f"Starting indexing for job {job_id} in folder {job_folder} with method {method}")
        JOB_STATUS[job_id] = {"status": "in_progress"}
        method_enum = IndexingMethod.Standard if method == "Standard" else IndexingMethod.Fast

        results = run_indexing_sync(job_folder , method_enum)
        errors = [r.errors for r in results if getattr(r, "errors", None)]

        if errors and any(errors):
            print(f"Indexing completed with errors for job {job_id}: {errors}")
            JOB_STATUS[job_id] = {"status": "error", "details": str(errors)}
        else:
            print(f"Indexing completed successfully for job {job_id}")
            JOB_STATUS[job_id] = {"status": "completed"}
    except Exception as e:
        print(f"Error in start_indexing for job {job_id}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        JOB_STATUS[job_id] = {"status": "error", "details": str(e)}
    except Exception as e:
        JOB_STATUS[job_id] = {
            "status": "error",
            "details": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }
