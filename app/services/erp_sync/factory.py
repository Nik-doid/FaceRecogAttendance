"""Factory for the ERP sync service + scheduler from settings.

The ERP sync is optional. When ``erp_sync_enabled`` is false (the default in tests and
local demo) a disabled service is returned that no-ops, keeping the control plane and
tests independent of a MySQL attendance DB.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.repositories.recognition_log_repo import RecognitionLogRepository
from app.services.erp_sync.client import ErpDbConfig, ErpMysqlClient
from app.services.erp_sync.mapping import CameraMapping, InOutResolver
from app.services.erp_sync.service import ErpSyncService
from app.storage.snapshot import SnapshotStorage


def build_erp_sync(
    settings: Settings, snapshot_storage: SnapshotStorage | None = None
) -> ErpSyncService:
    client = ErpMysqlClient(
        ErpDbConfig(
            host=settings.erp_db_host,
            port=settings.erp_db_port,
            database=settings.erp_db_name,
            user=settings.erp_db_user,
            password=settings.erp_db_password,
            employee_table=settings.erp_employee_table,
            employee_code_column=settings.erp_employee_code_column,
            employee_id_column=settings.erp_employee_id_column,
            employee_active_filter=settings.erp_employee_active_filter,
        )
    )
    return ErpSyncService(
        repo=RecognitionLogRepository(),
        client=client,
        camera_mapping=CameraMapping(settings.erp_camera_mapping),
        in_out_resolver=InOutResolver(settings.erp_in_out_mode),
        verify_mode=settings.erp_verify_mode,
        created_by=settings.erp_created_by,
        batch_size=settings.erp_sync_batch_size,
        enabled=settings.erp_sync_enabled,
        snapshot_storage=snapshot_storage,
    )
