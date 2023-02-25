import DropDowns from './dropdown'
import ButtonU from './button'

const LargeNavBar = () => {
    return (
        <>
            <div className="grid grid-rows-1 grid-flow-col auto-cols-max justify-between bg-white">
                <ButtonU name="HOME" />
                <DropDowns
                    name="TEAM"
                    arr={['MEMBERS', 'ATTRIBUTIONS', 'COLLABORATIONS']}
                />
                <DropDowns
                    name="PROJECT"
                    arr={[
                        'DESCRIPTION',
                        'DESIGN',
                        'ENGINEERING',
                        'RESULTS',
                        'IMPLEMENTATION',
                        'CONTRIBUTION',
                        'NOTEBOOK',
                        'SAFETY'
                    ]}
                />
                <DropDowns
                    name="MODEL"
                    arr={['GENE SEQUENCE', 'CIRCULAR DNA']}
                />
                <DropDowns name="PARTS" arr={['PARTS COLLECTION']} />
                <DropDowns
                    name="HUMAN PRACTICES"
                    arr={[
                        'INTEGRATED HUMAN PRACTICES',
                        'EDUCATIONAL OUTREACH',
                        'PATNERSHIP'
                    ]}
                />
                <ButtonU name="JUDGING FORM" />
            </div>
        </>
    )
}

export default LargeNavBar
