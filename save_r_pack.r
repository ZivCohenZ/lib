library(downloader)

library(miniCRAN)
setwd("F:/kk/r")

library("miniCRAN")


# pkgs <- c('stringr', 'devtools', 'ggplot2', 'dplyr')
pkgs <- c('sqldf','dplyr')


revolution <- c(CRAN="http://cran.revolutionanalytics.com")
pkgList <- pkgDep(pkgs, repos=revolution, type="source" )


pkgInfo <- download.packages(pkgs = pkgList, destdir = getwd(), type = "win.binary")
write.csv(file = "pkg.csv", basename(pkgInfo[, 2]), row.names = FALSE)
