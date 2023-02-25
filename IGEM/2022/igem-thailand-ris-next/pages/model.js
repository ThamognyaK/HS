function Model({}) {
    return (
        <div>
            <div class="titleScreen1">
                <h1 class="titlescreentext">MODELING</h1>
            </div>
            <div class="body">
                <div class="sizedbox40" />
                <h1 class="header1">Biopanning</h1>
                <div class="sizedbox20"></div>
                <p class="indented1">
                    The indirect ELISA process that screens for a nanobody that
                    binds best with the target bacteria, resulted in a clone
                    that did not have any genes. A potential reason could be
                    that the nanobody gene sent by Chulalongkorn University only
                    had a vector, thus no protein coding sequence could be
                    identified. Though, if more clones than just the twenty sent
                    were screened, a nanobody with a gene and sequence would
                    have been identified. Furthermore, the prepared phage
                    library was old, therefore, the chances of identifying a
                    nanobody gene was even lower.
                </p>
                <div class="sizedbox20"></div>
                <h1 class="header1">Modified Bombolitin Modeling</h1>
                <div class="sizedbox20"></div>
                <p class="indented1">
                    The bombolitin antimicrobial peptide is already established
                    to be an antimicrobial peptide that has the ability to
                    penetrate cell membranes and cause cell lysis. Our project
                    aims to modify the structure of the bombolitin through the
                    addition of three amino acids in order to increase the
                    peptide&#39;s hydrophobicity and ability to penetrate the{' '}
                    <i>Ralstonia Solanacearum</i> cell membranes. There are
                    various possibilities in which the three additional amino
                    acids can be ordered. Although it is possible to express and
                    perform in-vitro testing for all the combinations, it is
                    much easier to check the most likely combinations through
                    modeling and molecular docking.
                </p>
                <div class="sizedbox20" />
                <h1 class="header1">Molecular Model and Docking</h1>
                <div class="sizedbox20"></div>
                <p class="indented1">
                    We utilized HADDOCK and Alphafold programs to model the
                    peptide and perform molecular docking. After modeling the
                    peptide structure, we end up with the target, bombolitin and
                    protein target, and cell membrane. From this, we are able to
                    find the specific bind, the type of cell protein, and
                    binding affinity of our modeled peptide modification.
                </p>
                <div class="sizedbox20" />
                <h1 class="header1">Results</h1>
                <div class="sizedbox20" />
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/screen-shot-2565-10-12-at-03-15-11.png"
                        class="fitimage1"
                    />
                    <p class="fitimage1">
                        Bombolitin with LLL Modification Structure Prediction
                    </p>
                </div>
                <div class="sizedbox20"></div>
                <h1 class="header1">Analysis</h1>
                <div class="sizedbox20"></div>
                <p class="indented1">
                    As a result of altering the structure of the bombolitin
                    through adding additional amino acids, the modeling and
                    calculations from molecular docking shows that the modified
                    version LLL is the structure with the highest efficiency.
                    The enthalpy levels of the LLL modification is the lowest
                    when compared to other versions. The more negative the
                    enthalpy, the higher the affinity the peptide has in regards
                    to being efficient in penetrating the membrane.
                </p>
                <div class="sizedbox20"></div>
                <h1 class="header1">Future Work</h1>
                <div class="sizedbox20" />
                <p class="indented1">
                    By adding three amino acids and changing the structure in
                    the program, the efficiency of the modified bombolitin is
                    determined by the given polarity, hydrophobicity, and
                    hydrophilicity. Visualizing the testing of the efficiency of
                    the novel design through modeling before experimentation was
                    very beneficial when taking the time constraints into
                    consideration. However, lab experimentation is very
                    essential to test the applications of the peptide. To
                    conclude, the binding energy and the model have been
                    utilized throughout the lab as reference.
                </p>
                <div class="sizedbox20"></div>
                <h1 class="header1">Reference</h1>
                <div class="sizedbox20" />
                <p class="indented1">
                    [1] Jumper, J et al. Highly accurate protein structure
                    prediction with AlphaFold. Nature (2021).
                </p>

                <div class="sizedbox40" />
            </div>
        </div>
    )
}

export default Model
