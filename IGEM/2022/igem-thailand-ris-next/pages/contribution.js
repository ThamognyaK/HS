function Contribution({ children }) {
    return (
        <div>
            <div class="titleScreen1">
                <h1 class="titlescreentext">CONTRIBUTION</h1>
            </div>
            <div class="sizedbox60" />
            <div class="body">
                <h1 class="header1">Contribution</h1>
                <div class="sizedbox20"></div>
                <p class="indented">
                    The final results from both biopanning and ELISA that we
                    received from Kasetsart University are arranged to be
                    provided for future iGEM teams. In our test to find the most
                    specific nanobody that binds with the solanacearum, we were
                    able to identify it out of the 20 PBS clones. We were able
                    to demonstrate more effective use of both TB1 and HB2151.
                    Future teams could implement this technique to make
                    biopanning even more effective. Another advantage of our
                    technique was the use of Tween-20 which acts as a detergent
                    to decrease background staining. The phosphate-buffered
                    saline solution that we used and calculated could be used
                    for future teams to get the highest yield of specific
                    nanobodies. Finally, our peptide can also be utilized by
                    future teams.
                </p>

                <div class="sizedbox40" />
                <h1 class="header1">Ligation Ratio</h1>
                <div class="sizedbox20"></div>
                <p class="indented">
                    Ligation is an important step when working with
                    transformation. Ligation is the process of incubating DNA
                    sequences together to allow for a vector and an insert to be
                    united. The ratio of ligation is also of the utmost
                    importance since it determines whether or not the vector and
                    insert will be able to unite with one another. In our case
                    at the lab, the initial ratio was 3:5 between vector to
                    insert, however, throughout cloning, we discovered that this
                    ratio did not work. This led us to try a different ratio of
                    1:2 between vector and insert, which eventually worked.
                </p>

                <div class="sizedbox40" />
                <h1 class="header1">
                    Characterizing an existing part BBa_J18932
                </h1>
                <div class="sizedbox20"></div>
                <p class="indented">
                    We contributed to future iGEM teams by completing an
                    experimental characterization on the existing part
                    <a
                        href="http://parts.igem.org/Part:BBa_J18932"
                        class="blue"
                    >
                        {' '}
                        <u>BBa_J18932</u>
                    </a>
                </p>
            </div>

            <div class="sizedbox60" />
        </div>
    )
}

export default Contribution
