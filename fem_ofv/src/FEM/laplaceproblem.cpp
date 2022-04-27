/*
 * ARL_Topologies - An extensible topology optimization program
 *
 * Written in 2022 by Raymond A. Wildman
 *
 * To the extent possible under law, the author(s) have dedicated all copyright
 * and related and neighboring rights to this software to the public domain
 * worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication along
 * with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 *
 */

#include "laplaceproblem.h"

LaplaceProblem::LaplaceProblem(const Topologies::TOMesh &inMesh, const Topologies::GenericMaterial &baseMat) : FEMProblem(inMesh, baseMat)
{
}

void LaplaceProblem::changeBoundaryConditionsTo(const std::vector<ExoBC> &bcVec)
{
    const std::size_t nnodes = probMesh->getNumUnknowns();
    fixedDOFs = std::vector<bool>(nnodes, false);
    loadVec.clear();
    for (auto const &curBC : bcVec)
    {
        for (auto kn : curBC.nodeIDVec)
        {
            // Set up supports (zero BC)
            if (curBC.isSupport)
            {
                fixedDOFs[kn] = curBC.xsup || curBC.ysup || curBC.zsup;
            }
            else // Set up loads (non-zero BC)
            {
                loadVec[kn] = curBC.loadVec.x + curBC.loadVec.y + curBC.loadVec.z;
                fixedDOFs[kn] = true;
            }
        }
    }
    // Save global row numbers for compressed system (without fixed dofs)
    bfRemapVec.resize(nnodes);
    std::size_t curBF = 0;
    for (std::size_t k = 0; k < fixedDOFs.size(); ++k)
    {
        if (!fixedDOFs[k])
        {
            bfRemapVec[k] = curBF++;
        }
    }
    numFreeDOFs = curBF;
    solveProblem();
}

void LaplaceProblem::solveProblem()
{
    // Set up vectors and matrices
    setMatrix();
    setVector();
    // Matrix stuff:
    // Save RHS to use for computation of compliance
    pForce = std::unique_ptr<EigenVector>(new EigenVector(*pVVec));
    invalid = false;
    // Solve
    Eigen::ConjugateGradient<EigenSparseMat, Eigen::Lower | Eigen::Upper> solver;
    solver.compute(*pFEMMatrix);
    solver.setTolerance(itTol);
    unsigned niters = 10000;
    solver.setMaxIterations(niters);
    *pVVec = solver.solve(*pForce);
    if (solver.iterations() >= niters)
        invalid = true;
}

void LaplaceProblem::setMatrix()
{
    std::vector<EigenT> sparseVec;
    sparseVec.reserve(10 * numFreeDOFs); // Size of this depends on matrix bandwidth, which is unknown as it depends on mesh connectivity
    std::size_t nelems = probMesh->getNumElements();
    for (std::size_t ielem = 0; ielem < nelems; ++ielem)
    {
        const Eigen::MatrixXd elemMat = probMesh->getLaplacianElemMat(ielem, 1.0);
        assembleMatrix(sparseVec, ielem, elemMat);
    }
    pFEMMatrix = std::unique_ptr<EigenSparseMat>(new EigenSparseMat(numFreeDOFs, numFreeDOFs));
    pFEMMatrix->setFromTriplets(sparseVec.begin(), sparseVec.end());
}

void LaplaceProblem::assembleMatrix(std::vector<EigenT> &rseMat, const std::size_t kelem, const Eigen::MatrixXd &elemMat) const
{
    const unsigned numElemNodes = probMesh->getNumElementNodes(kelem);
    for (unsigned iTst = 0; iTst < numElemNodes; iTst++)
    {
        const std::size_t gTst = probMesh->getGlobalBF(kelem, iTst);
        for (unsigned iBas = 0; iBas < numElemNodes; iBas++)
        {
            const std::size_t gBas = probMesh->getGlobalBF(kelem, iBas);
            const bool isFree = !(fixedDOFs[gTst] || fixedDOFs[gBas]);
            if (fabs(elemMat(iTst, iBas)) > 1.e-16 && isFree)
            {
                rseMat.push_back(EigenT(bfRemapVec[gTst],
                                        bfRemapVec[gBas],
                                        elemMat(iTst, iBas)));
            }
        }
    }
}

void LaplaceProblem::setVector()
{
    pVVec = std::unique_ptr<EigenVector>(new EigenVector(numFreeDOFs));
    pVVec->setZero();
    const std::size_t numNodes = probMesh->getNumUnknowns();
    std::size_t nelems = probMesh->getNumElements();
    for (std::size_t ielem = 0; ielem < nelems; ++ielem)
    {
        const Eigen::MatrixXd elemMat = probMesh->getLaplacianElemMat(ielem, 1.0);
        assembleVector(ielem, elemMat);
    }
}

void LaplaceProblem::assembleVector(const std::size_t kelem, const Eigen::MatrixXd &elemMat) const
{
    const unsigned numElemNodes = probMesh->getNumElementNodes(kelem);
    for (unsigned iTst = 0; iTst < numElemNodes; iTst++)
    {
        const std::size_t gTst = probMesh->getGlobalBF(kelem, iTst);
        for (unsigned iBas = 0; iBas < numElemNodes; iBas++)
        {
            const std::size_t gBas = probMesh->getGlobalBF(kelem, iBas);
            const auto loadIter = loadVec.find(gBas);
            if (loadIter != loadVec.end() && !fixedDOFs[gTst])
            {
                (*pVVec)(bfRemapVec[gTst]) -= loadIter->second * elemMat(iTst, iBas);
            }
        }
    }
}

std::pair<double, bool> LaplaceProblem::computeCompliance()
{
    if (invalid)
    {
        std::cout << "Warning, FEM OFV not valid" << std::endl;
        return std::pair<double, bool>(1e6, false);
    }
    double sum = pForce->adjoint() * (*pVVec);
    return std::pair<double, bool>(sum, true);
}

double LaplaceProblem::elementCompliance(const std::size_t kelem, const EigenDenseMat &elemMat) const
{
    double res = 0.;
    const unsigned numElemNodes = probMesh->getNumElementNodes(kelem);
    for (unsigned iTst = 0; iTst < numElemNodes; iTst++)
    {
        const std::size_t gTst = probMesh->getGlobalBF(kelem, iTst);
        for (unsigned iBas = 0; iBas < numElemNodes; iBas++)
        {
            const std::size_t gBas = probMesh->getGlobalBF(kelem, iBas);
            if (!(fixedDOFs[gTst] || fixedDOFs[gBas]))
                res += elemMat(iTst, iBas) * (*pVVec)(bfRemapVec[gTst]) * (*pVVec)(bfRemapVec[gBas]);
        }
    }
    return res;
}

std::vector<double> LaplaceProblem::gradCompliance(const Topologies::TOMesh &inMesh) const
{
    const std::size_t nelems = probMesh->getNumElements();
    // Precompute uku values
    std::vector<double> ukuVec(nelems);
    for (std::size_t ielem = 0; ielem < nelems; ++ielem)
    {
        const Eigen::MatrixXd elemMat = probMesh->getElementMatrix(ielem);
        ukuVec[ielem] = -elementCompliance(ielem, elemMat);
        ukuVec[ielem] /= inMesh.getOptVal(ielem);
    }
    return ukuVec;
}
