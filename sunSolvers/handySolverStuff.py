import numpy as np
from scipy.sparse import diags, spmatrix


def centralDifferencesMatrix(steps: np.ndarray) -> spmatrix:
    """
    Given an array of steps (i.e. x_i) at which a function f is evaluated (i.e. f_i) returns a tridiagonal matrix such that Af are the central differences of f
    On edges the differences are approximated by one sided differences

    for uniform stepsize h

              ( -1   1   0   0   0  0 )
              (-1/2  0  1/2  0   0  0 )
    A = 1/h * (  0 -1/2  0  1/2  0  0 )
              (  0   0 -1/2  0  1/2 0 )
              (  0   0   0   0  -1  1 )

    """
    stepsizes = np.diff(steps)
    numberOfSteps = len(steps)

    underDiag = -1 / (stepsizes[:-1] + stepsizes[1:])
    underDiag = np.append(underDiag, -1 / stepsizes[-1])
    overDiag = 1 / (stepsizes[:-1] + stepsizes[1:])
    overDiag = np.insert(overDiag, 0, 1 / stepsizes[0])
    diag = np.zeros(numberOfSteps)
    diag[0] = -1 / stepsizes[0]
    diag[-1] = 1 / stepsizes[-1]

    centeredDifferences = diags(
        [underDiag, diag, overDiag], [-1, 0, 1], shape=(numberOfSteps, numberOfSteps)  # type: ignore I'm unsure why the [-1, 0, 1] is throwing an error, this is literally the example from the documentation
    )
    return centeredDifferences


def forwardDifferencesMatrix(steps: np.ndarray) -> spmatrix:
    """
    Given an array of steps (i.e. x_i) at which a function f is evaluated (i.e. f_i) returns a tridiagonal matrix such that Af are the forward differences of f
    On edges the differences are approximated by one sided differences

    for uniform stepsize h

              ( -1   1   0   0   0  )
              (  0  -1   1   0   0  )
    A = 1/h * (  0   0  -1   1   0  )
              (  0   0   0  -1   1  )
              (  0   0   0  -1   1  )
    """

    dx = np.diff(steps)
    numberOfSteps = len(steps)

    underDiag = np.zeros(numberOfSteps - 1)
    underDiag[-1] = -1 / dx[-1]
    diag = np.append(-1 / dx, 1 / dx[-1])
    overDiag = 1 / dx

    forwardDifferences = diags([underDiag, diag, overDiag], [-1, 0, 1], shape=(numberOfSteps, numberOfSteps))  # type: ignore I'm unsure why the [-1, 0, 1] is throwing an error, this is literally the example from the documentation
    return forwardDifferences


def backwardDifferencesMatrix(steps: np.ndarray) -> spmatrix:
    """
    Given an array of steps (i.e. x_i) at which a function f is evaluated (i.e. f_i) returns a tridiagonal matrix such that Af are the backward differences of f
    On edges the differences are approximated by one sided differences

    for uniform stepsize h

              ( -1   1   0   0   0  )\n
              ( -1   1   0   0   0  )\n
    A = 1/h * (  0  -1   1   0   0  )\n
              (  0   0  -1   1   0  )\n
              (  0   0   0  -1   1  )\n
    """
    dx = np.diff(steps)
    numberOfSteps = len(steps)

    overDiag = np.zeros(numberOfSteps - 1)
    overDiag[0] = 1 / dx[0]
    diag = np.insert(1 / dx, 0, -1 / dx[0])
    underDiag = -1 / dx

    backwardDifferences = diags([underDiag, diag, overDiag], [-1, 0, 1], shape=(numberOfSteps, numberOfSteps))  # type: ignore I'm unsure why the [-1, 0, 1] is throwing an error, this is literally the example from the documentation
    return backwardDifferences


def secondCentralDifferencesMatrix(
    steps: np.ndarray, constantBoundaries=False
) -> spmatrix:
    """Given an array of steps (i.e. x_i) at which a function f is evaluated (i.e. f_i) returns a tridiagonal matrix such that Af are the second central differences of f

    Args:
        steps (np.ndarray): array of steps, i.e. x_(i+1) - x_i
        constantBoundaries (bool, optional): whether to use constant boundary conditions, i.e. identity at x_0 and x_n. Defaults to False.

    Returns:
        spmatrix: _description_
    """
    forward = forwardDifferencesMatrix(steps)
    backward = backwardDifferencesMatrix(steps)

    secondCentral = forward.dot(backward)
    # becuase of the nature of the one sided differences
    # and their behaviour at the edges
    # and the order of the matrix multiplication above
    # the first row gets obliterated
    if constantBoundaries:
        const = np.zeros(len(steps))
        const[0] = 1
        secondCentral[-1, :] = const[::-1]
        secondCentral[0, :] = const[:]
    else:
        secondCentral[0, :] = secondCentral[1, :]

    return secondCentral


if __name__ == "__main__":
    a = secondCentralDifferencesMatrix(np.arange(8), constantBoundaries=True)
    b = a.toarray()
    print(b)
    pass
