


if (n > 2)
{
	sequence = seq(2, n);
	primes_matrix = list();
	for (i in seq(2, n))
	{
		if (any(sequence == i))
		{
			primes = list(primes, i);
			s = list(sequence[(sequence %% i)] != 0, i);
		}
	}
	return (primes);
}
else
{
	stop;
}
