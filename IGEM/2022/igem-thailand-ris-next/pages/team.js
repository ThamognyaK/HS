import Link from 'next/link'

const Team = ({}) => {
    return (
        <div>
            <div class="navbg">
                <div class="column body">
                    <div class="sizedbox100" />
                    <div class="body">
                        <h1 class="title">TEAM</h1>
                    </div>
                    <div class="sizedbox40" />
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/icons8-chevron-100.png"
                        width={80}
                    />
                    <div class="sizedbox40" />
                    <Link href="/members">
                        <h1 class="center-text hover-underline-animation1">
                            MEMBERS
                        </h1>
                    </Link>
                    <div class="sizedbox20" />
                    <Link href="/attributions">
                        <h1 class="center-text hover-underline-animation1">
                            ATTRIBUTIONS
                        </h1>
                    </Link>
                    <div class="sizedbox20" />
                    <Link href="/collaborations">
                        <h1 class="center-text hover-underline-animation1">
                            COLLABORATIONS
                        </h1>
                    </Link>
                    <div class="sizedbox20" />
                </div>
            </div>
        </div>
    )
}

export default Team
