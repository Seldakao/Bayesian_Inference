library(rethinking)
# Linear Models -----------------------------------------------------------

# simulate coin-flip (random walk)
# 1000 people line up in the middle of the ground and flip coins for 16 times, if head, move to the right, if tail move to the left
# simulate the final position of each person
pos <- replicate(1000, sum(runif(16, -1, 1)))
hist(pos)
plot(density(pos))

# normal by multiplication
# simulate 12 numbers between 1.0 and 1.1 (growth rate) and multiply them
prod(1 + runif(12, 0, 0.1))

#10000 random products (small effects that muliply together are approximately additive)
growth <- replicate(10000, prod(1 + runif(12, 0, 0.1)))
dens(growth, norm.com = TRUE)

# bigger amount
big <- replicate(10000, prod(1 + runif(12, 0, 0.5)))
small <- replicate(10000, prod(1 + runif(12, 0, 0.01)))
dens(big)
dens(small)

# normal by log-muliplication
log.big <- replicate(10000, log(prod(1 + runif(12, 0, 0.5))))
dens(log.big)

##  Guassian Distribution
# probability density can be greater than 1, but probability mass cannot
dnorm(0, 0, 0.1)

# A Gaussain model of height

data(Howell1)
d <- Howell1
# height column
dens(d$height)
# first variable
d[[1]]

# filter the data frame by age (height is strongly correlated with age before adulthood)
# specific rows with all columns
d2 <- d[d$age >18 ,]

# accessing d[row, col]
d2[2,3]
# distribution of height
dens(d2$height)


# Grid approximation of the posterior distribution - Linear Model ---------

mu.list <- seq(from = 140, to = 160, length.out=200)
sigma.list <- seq(from = 4, to =9, length.out =200)
post <- expand.grid(mu = mu.list, sigma = sigma.list)

post$LL <- sapply(1:nrow(post),
                  function(i) sum(
                    dnorm(
                      d2$height,
                      mean = post$mu[i],
                      sd = post$sigma[i],
                      log = TRUE
                    )
                  ))

post$prod <- post$LL + dnorm(post$mu, 178, 20, TRUE) + dunif(post$sigma, 0, 50, TRUE)
post$prob <- exp(post$prod - max(post$prod))

contour_xyz(post$mu, post$sigma, post$prob)
image_xyz(post$mu, post$sigma, post$prob)


# Sampling from the posterior ---------------------------------------------

sample.rows <- sample(1:nrow(post), size =1e5, replace = TRUE, prob = post$prob)
sample.mu <- post$mu[sample.rows]
sample.sigma <- post$sigma[sample.rows]

plot(sample.mu, sample.sigma, cex = 0.5, pch = 16, col = col.alpha(rangi2, 0.1))

dens(sample.mu)
dens(sample.sigma)

HPDI(sample.mu)
HPDI(sample.sigma)


# Sample Size and the Normality of Sigma Posterior ------------------------

d3 <- sample(d2$height, size = 20)
mu.list <- seq(from = 140, to = 160, length.out=200)
sigma.list <- seq(from = 4, to =9, length.out =200)
post2 <- expand.grid(mu = mu.list, sigma = sigma.list)

# go through each mu/sigma pair in the grod and sum the density of the actual heights from each normal distribution
post2$LL <- sapply(1:nrow(post2),
                  function(i) sum(
                    dnorm(
                      d3,
                      mean = post2$mu[i],
                      sd = post2$sigma[i],
                      log = TRUE
                    )
                  ))

# getting posterior (addition here because it is log scale)
post2$prod <- post2$LL + dnorm(post2$mu, 178, 20, TRUE) + dunif(post2$sigma, 0, 50, TRUE)
# reverse the log
post2$prob <- exp(post2$prod - max(post2$prod))

# randome sample the rows according to their probability
sample2.rows <- sample(1:nrow(post), size =1e4, replace = TRUE, prob = post$prob)
# get the corresponding mu
sample2.mu <- post2$mu[sample2.rows]
# get the corresponding sigma
sample2.sigma <- post2$sigma[sample2.rows]

plot(sample2.mu, sample2.sigma, cex = 0.5, pch = 16, col = col.alpha(rangi2, 0.1))

dens(sample2.sigma, norm.com = TRUE)
dens(sample2.mu, norm.com = TRUE)



# Fitting the model with map ----------------------------------------------

# data was retrived before

flist <- alist(
  height ~ dnorm(mu, sigma),
  mu ~ dnorm(178, 20),
  sigma ~dunif(0, 50)
)

m4.1 <- map(flist, data = d2)
precis(m4.1, prob = 0.95)

# starting values for MAP
start <- list(
  mu = mean(d2$height),
  sigma = sd(d2$height)
)

# narrow prior
m4.2 <- map(
  alist(
    height ~dnorm(mu, sigma),
    mu ~ dnorm(178, 0.1),
    sigma ~ dunif(0, 50)
    ),
  data = d2
)
precis(m4.2)

# Sampling from a map fit
vcov(m4.1)

diag(vcov(m4.1))
cov2cor(vcov(m4.1))

post <- extract.samples(m4.1, n = 1e4)
head(post)
# will mean of the sampling from the posterior close to MAP?
precis(post)

plot(post)

library(MASS)
post <- mvrnorm(n = 1e4, mu = coef(m4.1), Sigma = vcov(m4.1))

# log scale of sigma
m4.1_logsigma <-map (
  alist(
    height ~ dnorm(mu, exp(log_sigma)),
    mu ~ dnorm(178, 20),
    log_sigma ~ dnorm(2, 10)
  ),
  data = d2
)

post <- extract.samples(m4.1_logsigma)
sigma <- exp(post$log_sigma)

dens(sigma)


# Adding a predictor ------------------------------------------------------

plot(d2$height ~ d2$weight)

# load data again, since it is a long way back
library(rethinking)
data(Howell1)
d <- Howell1
d2 <- d[d$age >= 18,]

#fit model 
M4.3 <- map(
  alist(
    height ~ dnorm(mu, sigma),
    mu <- a +b * weight,
    a ~ dnorm(156, 100),
    b ~ dnorm( 0, 10),
    sigma ~ dunif(0, 50)
  ),
  data = d2
)

m4.3 <- map(
  alist(
    height ~ dnorm( mu, sigma),
    mu <- a + b* weight,
    a ~ dnorm(178, 100),
    b ~ dnorm( 0, 10),
    sigma ~ dunif(0, 50)
  ),
  data = d2
)

precis(m4.3)
# covariance variance matrix
precis(m4.3, corr = TRUE)
cov2cor(vcov(m4.3))

# centering to avoid correlation between parameters?
d2$weight.c <- d2$weight -mean(d2$weight)
plot(d2$height ~ d2$weight.c)

m4.4 <- map(
  alist(
    height ~ dnorm(mu, sigma),
    mu <- a + b*weight.c,
    a ~ dnorm(178, 100),
    b ~ dnorm(0, 10),
    sigma ~ dunif(0,50)
  ),
  data = d2
)

precis(m4.4, corr = TRUE)

plot(height ~ weight, data = d2)
abline(a = coef(m4.3)["a"], b = coef(m4.3)["b"])

post <- extract.samples(m4.3)
post[1:5,]

# use a small sample and estimate the model
N <- 352
dN <- d2[1:N,]
mN <- map(
  alist(
    height ~dnorm(mu, sigma),
    mu <- a +b*weight,
    a ~ dnorm(178, 100),
    b ~ dnorm (0, 10),
    sigma ~ dunif(0, 50)
  ),
  data = dN
)

# extract 20 samples from the posterior
post <- extract.samples(mN, n=20)

# display raw data and sample size
plot(dN$weight, dN$height,
     xlim = range(d2$weight), ylim = range(d2$height),
col = rangi2, xlab = "weight", ylab = "height")
mtext(concat("N =", N))

# plot the lines, with transparency
for (i in 1:20)
    abline(a = post$a[i], b = post$b[i], col = col.alpha("black", 0.3))  

# ploting regression intervals and contoturs

post <- extract.samples(mN, n= 10000)
mu_at_50 <-post$a + post$b * 50

dens(mu_at_50, col = rangi2, lwd = 2, xlab = "mu|weight = 50")

HPDI(mu_at_50, prob = 0.89)

mu <- link(m4.3, data = data.frame(weight = d2$weight))
str(mu)

# define sequence of weights to compute predictions for
# these vaues will be on the horizontal axis

weight.seq <- seq(from =25, to  = 70, by = 1)
# use link to compute mu
# for each sample from posterior
# and for each weight in weight.seq
mu <- link(m4.3, data = data.frame(weight = weight.seq))
str(mu)

# use type ="n" to hide raw data
plot(height ~ weight, d2)

for (i in 501:600)
    points(weight.seq, mu[i,], pch = 20, col= col.alpha(rangi2,0.4))

# summarize the distribution of mu
mu.mean <- apply(mu, 2, mean)
mu.HPDI <- apply(mu, 2, HPDI, prob =0.89)

#plot raw data
# fading out points to make line and interval more visiable
plot(height ~ weight, data = d2, col = col.alpha(rangi2, 0.5))

#plot the MAP lline, aka the mean mu for each weight
lines(weight.seq, mu.mean)

#plot a shaded region for %89 HPDI
shade(mu.HPDI, weight.seq)

### prediction interval
sim.height <- sim(m4.3, data = list(weight = weight.seq), n = 1e4)
str(sim.height)

#summarise
height.PI <- apply(sim.height, 2, PI, prob =0.89)

#plot the raw data
plot(height ~ weight, de, col = col.alpha(rangi2, 0.5))

# draw MAP line
lines(weight.seq, mu.mean)

#draw HPDI region for the line
shade(mu.HPDI, weight.seq)

# draw PI region for simulated heights
shade(height.PI, weight.seq)

## How does sim work?
post <- extract.samples(m4.3)
weight.seq <- 25:70
sim.height <- sapply(weight.seq, function(weight)
  rnorm(
    n = nrow(post),
    mean = post$a + post$b*weight,
    sd = post$sigma
  )
  )
height.PI <- apply(sim.height, 2, PI, prob =0.89)


# Polynomial regression ---------------------------------------------------
library(rethinking)
data(Howell1)
d <- Howell1
str(d)

#standardise the weights
d$weight.s <- (d$weight -mean(d$weight))/sd(d$weight)

plot(d$weight.s, d$height)

d$weight.s2 <- d$weight.s^2
m4.5 <- map(
  alist(
    height ~ dnorm(mu, sigma),
    mu <- a + b1*weight.s + b2*weight.s2,
    a ~ dnorm(178, 100),
    b1 ~ dnorm(0, 10),
    b2 ~ dnorm(0, 10),
    sigma ~ dunif(0, 50)
  ),
  data = d
)

precis(m4.5)

# plots
weight.seq <- seq(from = -2.2, to = 2, length.out = 30)
pred_dat <- list(weight.s = weight.seq, weight.s2 = weight.seq^2)
mu <- link(m4.5, data = pred_dat)
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI, prob = 0.89)
sim.height <- sim(m4.5, data = pred_dat)
height.PI <- apply(sim.height, 2, PI, prob = 0.89)

plot(height ~ weight.s, d, col = col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.PI, weight.seq)
shade(height.PI, weight.seq)

## cude
d$weight.s3 <- d$weight.s^3
m4.6 <- map(
  alist(
    height ~ dnorm(mu, sigma),
    mu <- a + b1*weight.s + b2*weight.s2 + b3*weight.s3,
    a ~ dnorm(178, 100),
    b1 ~ dnorm(0, 10),
    b2 ~ dnorm(0, 10),
    b3 ~ dnorm(0, 10),
    sigma ~ dunif(0, 50)
  ),
  data = d 
)

#plots
weight.seq <- seq(from = -2.2, to = 2, length.out = 30)
pred_dat <- list(weight.s = weight.seq, weight.s2 = weight.seq^2, weight.s3 = weight.seq^3)
mu <- link(m4.6, data = pred_dat)
mu.mean <- apply(mu, 2, mean)
mu.PI <- apply(mu, 2, PI, prob = 0.89)
sim.height <- sim(m4.6, data = pred_dat)
height.PI <- apply(sim.height, 2, PI, prob = 0.89)

plot(height ~ weight.s, d, col = col.alpha(rangi2, 0.5))
lines(weight.seq, mu.mean)
shade(mu.PI, weight.seq)
shade(height.PI, weight.seq)