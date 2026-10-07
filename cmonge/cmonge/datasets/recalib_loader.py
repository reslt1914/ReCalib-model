from __future__ import annotations

import scanpy as sc

from cmonge.datasets.single_loader import SciPlexModule


class ReCalibSciPlexModule(SciPlexModule):
    """
    SciPlex loader for the ReCalib-adapted dataset.

    Expression loading, control/target splitting,
    AE reduction, batching and sampling are inherited
    unchanged from the original CMonge SciPlexModule.

    Only the optional rank_genes_groups lookup is skipped,
    because ReCalib uses CMonge-safe hashed drug IDs.
    """

    def loader(self) -> None:
        if self.parent is not None:
            self.adata = self.parent
        else:
            self.adata = sc.read_h5ad(
                self.file_path
            )

        self.marker_genes = []
        self.marker_idx = []
        self.gene_idx_to_enum = {}
