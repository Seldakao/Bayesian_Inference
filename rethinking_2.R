# Grid Approximation  -----------------------------------------------------

# define grid
p_grid <- seq(from = 0, to = 1, length.out = 20)

# define prior
prior1 <- rep(1, 20)

# different prior 
prior2 <- ifelse(p_grid < 0.5 ,0, 1)
prior3 <- exp(-5*abs(p_grid-0.5))



# compute likelihood at each value in grid
likelihood <- dbinom(6, size = 9, prob = p_grid)

# compute product of likelihood anad prior
unstd.posterior1 <- likelihood * prior1
unstd.posterior2 <- likelihood * prior2
unstd.posterior3 <- likelihood * prior3
# standardize the posterior, so it shums to 1
posterior1 <- unstd.posterior1/sum(unstd.posterior1)
posterior2 <- unstd.posterior2/sum(unstd.posterior2)
posterior3 <- unstd.posterior3/sum(unstd.posterior3)

#
par(mfrow=c(3,1))
plot(p_grid, posterior1, col = "blue",type = "b", xlab = "probability of water", ylab = "posterior probability", sub = "prior 1")

plot(p_grid, posterior2, col = "green",type = "b", xlab = "probability of water", ylab = "posterior probability", sub = "prior 2")

plot(p_grid, posterior3, col = "red",type = "b", xlab = "probability of water", ylab = "posterior probability", sub = "prior 3")
mtext("20 points")

# Quadratic Approximation -------------------------------------------------


library(rethinking)
# change the number of size and data
w <- 24
n <- 36

globe.qa <-map(
  alist(
    w ~ dbinom(36 ,p), # binomail likelihood
    p ~ dunif(0,1)   # uniform prior
  ),
data = list(w=24))

# display summary of quadratic approximation
precis(globe.qa)
output <- attr(precis(globe.qa), 'output')
mean <- output$Mean
std <- output$StdDev

# analytical calculation
curve( dbeta(x, w+1, n-w+1), from = 0, to = 1)
# quadratic approximation
curve( dnorm(x, mean, std), lty = 2, add = TRUE)



