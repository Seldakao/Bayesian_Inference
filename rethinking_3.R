# Sampling from a grid-approximate posterior ------------------------------

p_grid <- seq(from = 0, to = 1, length.out = 1000)
prior = rep(1, 1000)
likelihood <- dbinom(6, size = 9, prob= p_grid)
posterior <- likelihood * prior
posterior <- posterior / sum(posterior)

samples <- sample(p_grid, prob = posterior, size = 1e5, replace = TRUE)
plot(samples)
dens(samples)

# add up posterior probability where p < 0.5
sum(posterior[p_grid<0.5])
# from the samples
sum(samples < 0.5)/1e5
sum(samples > 0.5 & samples <0.75)/1e5
# percentile intervals

## Another data WWW
p_grid <- seq(from = 0, to = 1, length.out = 1000)
prior <- rep(1,1000)
likelihood <- dbinom(3, size = 3, prob = p_grid)
posterior <- likelihood * prior
posterior <- posterior / sum(posterior)
samples <- sample(p_grid, size = 1e4, replace = TRUE, prob = posterior)
dens(samples)
# this can be misleading, because it excludes the most plausible parameter
PI(samples, prob =0.5)
quantile(samples, c(0.1, 0.9))
# Highest Posterior Density Interval
HPDI(samples, prob =0.5)


# Point Estimate ----------------------------------------------------------
# maximum a posteriori
p_grid[which.max(posterior)]
## mode of 
chainmode(samples, adj =0.01)
mean(samples)
median(samples)

# loss function
guess = 0.5
sum(posterior * abs(guess -p_grid))

# similar to lambda function and map in python?
# absolute loss
loss <- sapply(p_grid, function(d) sum(posterior * abs(d - p_grid)))
p_grid[which.min(loss)]


# Sampling to simulate prediction -----------------------------------------
# globe tossing for 2 times
dbinom(0:2, size = 2, prob =0.7)
rbinom(1, size = 2, prob = 0.7)
rbinom(10, size = 2, prob = 0.7)

# generate 100,000 dummy observations
dummy_w <-rbinom(1e5, size = 2, prob =0.7)
table(dummy_w)/1e5

# 9 tosses
dummy_w <- rbinom(1e5, size = 9, prob =0.7)
simplehist(dummy_w, xlab = "dummy water count")

# To simulate predicted observations for a single value of p, p =0.6
w <- rbinom(1e4, size = 9, prob =0.6)
simplehist(w)
# replace prob with sample from posterior
w <- rbinom(1e4, size = 9, prob = samples)
