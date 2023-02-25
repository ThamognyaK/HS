const HomePage = ({ children }) => {
    return (
        <div>
            <div class="titleScreen">
                <div class="column">
                    <div class="sizedbox100" />
                    <h1 class="title">THAILAND_RIS</h1>
                    <h3 class="description">
                        Modifying and expressing a novel antimicrobial peptide,
                        bombolitin, to combat bacterial wilt caused by the plant
                        pathogenic bacteria <i>Ralstonia solanacearum</i> in
                        solanaceous crops.
                    </h3>
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/igemlogopurple.png"
                        class="titlelogo"
                    />
                </div>
            </div>
            <div class="sizedbox100" />
            <h1 class="heading" align="center">
                PROMOTIONAL VIDEO
            </h1>
            <div class="stack">
                <img
                    src="https://static.igem.wiki/teams/4314/wiki/blob-1.svg"
                    class="stackedchild bgblob"
                />
                <iframe
                    title="Thailand_RIS: Promotional Video (2022) [English]"
                    src="https://video.igem.org/videos/embed/76e51868-d703-4813-85d8-3f2d95c6f80a"
                    frameborder="0"
                    allowfullscreen=""
                    sandbox="allow-same-origin allow-scripts allow-popups"
                    class="video promovid"
                />
            </div>
            <div class="sizedbox40" />
        </div>
    )
}

export default HomePage
