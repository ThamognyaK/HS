const MemberStuff = (props: any) => {
    return (
        <div>
            <div className="member-row">
                <div className="member-column">
                    <h2 className="membertag text-lg">
                        {props.name1.toUpperCase()}
                    </h2>
                    <h5 className="membertag text-md">
                        {props.person1.toUpperCase()}
                    </h5>
                    <div className="row1">
                        <img src={props.img1} className="memberimage" />
                        <p className="member-text">{props.desc1}</p>
                    </div>
                </div>

                <div className="member-column-right">
                    <h2 className="membertag align-right text-lg">
                        {props.name2.toUpperCase()}
                    </h2>
                    <h5 className="membertag align-right text-md">
                        {props.person2.toUpperCase()}
                    </h5>
                    <div className="row1">
                        <p className="member-text align-right">{props.desc2}</p>
                        <img src={props.img2} className="memberimage" />
                    </div>
                </div>
            </div>
            <div className="sizedbox20" />
        </div>
    )
}

export default MemberStuff
