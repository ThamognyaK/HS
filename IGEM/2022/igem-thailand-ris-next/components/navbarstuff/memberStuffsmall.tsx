const MemberStuffS = (props: any) => {
    return (
        <div>
            <div className="column left">
                <h2 className="membertag1">{props.name1.toUpperCase()}</h2>
                <h5 className="membertag">{props.person1.toUpperCase()}</h5>
                <div className="row">
                    <img src={props.img1} className="memberimage" />
                    <p className="member-text">{props.desc1}</p>
                </div>
            </div>
            <div className="sizedbox30" />
            <div className="column right">
                <h2 className="membertag1 right">
                    {props.name2.toUpperCase()}
                </h2>
                <h5 className="membertag right">
                    {props.person2.toUpperCase()}
                </h5>
                <div className="row">
                    <p className="member-text">{props.desc2}</p>
                    <img src={props.img2} className="memberimage" />
                </div>
            </div>
        </div>
    )
}

export default MemberStuffS
