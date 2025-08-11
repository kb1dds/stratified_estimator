# Just for fun, let's umap the lollipop

library(tidyverse)
library(umap)
dat <- read_csv('lollipop_example.csv')
lolli.umap <- umap(dat[,1:3])
lolli.umap$layout |> 
  as_tibble() |> 
  bind_cols(dat) |> 
  mutate(candy = x < 0) |> 
  ggplot(aes(x=V1,y=V2,color=x, shape=candy)) + 
  geom_point()