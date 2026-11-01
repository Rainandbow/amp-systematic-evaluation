# Released ampir mature inference on new, hash-verified inputs. No model fitting.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L || !args[1] %in% c('check', 'smoke', 'full')) {
  stop('Usage: Rscript predict_ampir.R check|smoke|full QUERY_TSV OUTPUT_DIR IDENTITY_JSON')
}
for (key in c('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) {
  do.call(Sys.setenv, setNames(list('2'), key))
}
suppressPackageStartupMessages(library(ampir))
suppressPackageStartupMessages(library(digest))
suppressPackageStartupMessages(library(jsonlite))
if (as.character(packageVersion('ampir')) != '1.1.0') stop('Need existing ampir 1.1.0')
sha <- function(p) digest::digest(file=p, algo='sha256', serialize=FALSE)
root <- system.file(package='ampir')
files <- list.files(root, recursive=TRUE, full.names=TRUE, all.files=TRUE)
files <- files[!file.info(files)$isdir]
inventory <- as.list(setNames(vapply(files, sha, character(1)), normalizePath(files)))
identity <- list(version='1.1.0', variant='mature', R=R.version.string, files=inventory)
if (args[1] == 'check') {
  jsonlite::write_json(identity, args[4], auto_unbox=TRUE, pretty=TRUE, digits=NA)
  cat('READ-ONLY TOOL CHECK PASSED: ampir\n')
  quit(status=0)
}
frozen <- jsonlite::read_json(args[4], simplifyVector=FALSE)
if (!identical(frozen$version, identity$version) || !identical(frozen$R, identity$R) ||
    !identical(frozen$files, inventory)) stop('ampir installed release changed after check')
query_sha <- sha(args[2])
expected <- read.delim(args[2], stringsAsFactors=FALSE, check.names=FALSE)
if (args[1] == 'smoke') expected <- head(expected[expected$split == 'validation', ], 10L)
if (!nrow(expected) || anyDuplicated(expected$sequence_id) ||
    !all(grepl('^[ACDEFGHIKLMNPQRSTVWY]{10,}$', expected$sequence))) stop('Invalid supported ampir inputs')
if (file.exists(args[3])) stop('Output directory exists; preserve it')
dir.create(args[3], recursive=TRUE)
query <- data.frame(seq_name=expected$sequence_id, seq_aa=expected$sequence, stringsAsFactors=FALSE)
native <- ampir::predict_amps(query, model='mature', min_len=10L, n_cores=1L)
if (nrow(native) != nrow(expected) || anyDuplicated(native$seq_name) ||
    !setequal(native$seq_name, expected$sequence_id)) stop('ampir ID coverage mismatch')
native <- native[match(expected$sequence_id, native$seq_name), , drop=FALSE]
if (!identical(as.character(native$seq_aa), expected$sequence)) stop('ampir sequence changed')
if (any(!is.finite(native$prob_AMP)) || any(native$prob_AMP < 0 | native$prob_AMP > 1)) stop('Invalid scores')
if (sha(args[2]) != query_sha) stop('Query changed')
for (i in seq_along(files)) if (sha(files[i]) != inventory[[i]]) stop('Installed ampir release changed')
out <- expected[c('sequence_id', 'sequence', 'group_id', 'split', 'label')]
out$probability <- native$prob_AMP
# A predeclared reference threshold, NOT an ampir-author universal decision rule.
out$prediction <- as.integer(out$probability >= 0.5)
options(digits=17)
write.table(out, file.path(args[3], 'predictions.tsv'), sep='\t', quote=FALSE, row.names=FALSE, na='NA')
write.table(native, file.path(args[3], 'native.tsv'), sep='\t', quote=FALSE, row.names=FALSE)
jsonlite::write_json(list(identity=identity, inference_only=TRUE, input_sha256=query_sha,
  threshold='0.5 benchmark reference; not official universal cutoff',
  domain_note='mature short-peptide model; long sequences are extrapolation'),
  file.path(args[3], 'run_manifest.json'), auto_unbox=TRUE, pretty=TRUE, digits=NA)
cat('PUBLISHED COMPLETE: ampir', args[1], nrow(out), '\n')
