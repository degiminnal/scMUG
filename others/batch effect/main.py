
import anndata
import scanpy as sc
import omicverse as ov

filename ="" # the name of adata file, in which "batch" is used to distinguish different batchs
adata = anndata.read_h5ad(filename)
adata.obs_names = adata.obs_names.astype(str)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=3)

adata=ov.pp.qc(adata,
              tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250},
              batch_key='batch')
ov.utils.store_layers(adata,layers='counts')

adata=ov.pp.preprocess(adata,mode='shiftlog|pearson',
                       n_HVGs=3000,batch_key='batch')
adata = adata[:, adata.var.highly_variable_features]

ov.pp.scale(adata)
ov.pp.pca(adata,layer='scaled',n_pcs=50)

adata_combat=ov.single.batch_correction(adata,batch_key='batch',
                                        methods='combat',n_pcs=50)

adata_combat=ov.single.batch_correction(adata,batch_key='batch',
                                        methods='harmony',n_pcs=50)

adata_scanorama=ov.single.batch_correction(adata,batch_key='batch',
                                        methods='scanorama',n_pcs=50)