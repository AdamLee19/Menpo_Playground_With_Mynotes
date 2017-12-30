""" Test data for testing the protocol manager with ABCs. """


from abc import ABCMeta

from traits.adaptation.api import PurePythonAdapter as Adapter


#### 'Power plugs' metaphor ###################################################

#### Protocols ################################################################

class UKStandard(object, metaclass=ABCMeta):
    pass

class EUStandard(object, metaclass=ABCMeta):
    pass

class JapanStandard(object, metaclass=ABCMeta):
    pass

class IraqStandard(object, metaclass=ABCMeta):
    pass

#### Implementations ##########################################################

class UKPlug(object):
    pass

UKStandard.register(UKPlug)

class EUPlug(object):
    pass

EUStandard.register(EUPlug)

class JapanPlug(object):
    pass

JapanStandard.register(JapanPlug)

class IraqPlug(object):
    pass

IraqStandard.register(IraqPlug)

class TravelPlug(object):

    def __init__(self, mode):
        self.mode = mode

#### Adapters #################################################################

# UK->EU
class UKStandardToEUStandard(Adapter):
    pass

EUStandard.register(UKStandardToEUStandard)

# EU->Japan
class EUStandardToJapanStandard(Adapter):
    pass

JapanStandard.register(EUStandardToJapanStandard)

# Japan->Iraq
class JapanStandardToIraqStandard(Adapter):
    pass

IraqStandard.register(JapanStandardToIraqStandard)

# EU->Iraq
class EUStandardToIraqStandard(Adapter):
    pass

IraqStandard.register(EUStandardToIraqStandard)

# UK->Japan
class UKStandardToJapanStandard(Adapter):
    pass

JapanStandard.register(UKStandardToJapanStandard)

# Travel->Japan
class TravelPlugToJapanStandard(Adapter):
    pass

JapanStandard.register(TravelPlugToJapanStandard)

# Travel->EU
class TravelPlugToEUStandard(Adapter):
    pass

EUStandard.register(TravelPlugToEUStandard)


#### 'Editor, Scriptable, Undoable' metaphor ##################################

class FileType(object):
    pass

class IEditor(object, metaclass=ABCMeta):
    pass

class IScriptable(object, metaclass=ABCMeta):
    pass

class IUndoable(object, metaclass=ABCMeta):
    pass


class FileTypeToIEditor(Adapter):
    pass

IEditor.register(FileTypeToIEditor)
IScriptable.register(FileTypeToIEditor)

class IScriptableToIUndoable(Adapter):
    pass

IUndoable.register(IScriptableToIUndoable)


#### Hierarchy example ########################################################

class IPrintable(object, metaclass=ABCMeta):
    pass

class Editor(object):
    pass

class TextEditor(Editor):
    pass

class EditorToIPrintable(Adapter):
    pass

IPrintable.register(EditorToIPrintable)

class TextEditorToIPrintable(Adapter):
    pass

IPrintable.register(TextEditorToIPrintable)


#### Interface hierarchy example ##############################################

class IPrimate(object, metaclass=ABCMeta):
    pass

class IHuman(IPrimate):
    pass

class IChild(IHuman):
    pass

class IIntermediate(object, metaclass=ABCMeta):
    pass

class ITarget(object, metaclass=ABCMeta):
    pass

class Source(object):
    pass

IChild.register(Source)

class IChildToIIntermediate(Adapter):
    pass

IIntermediate.register(IChildToIIntermediate)

class IHumanToIIntermediate(Adapter):
    pass

IIntermediate.register(IHumanToIIntermediate)

class IPrimateToIIntermediate(Adapter):
    pass

IIntermediate.register(IPrimateToIIntermediate)

class IIntermediateToITarget(Adapter):
    pass

ITarget.register(IIntermediateToITarget)


#### Non-trivial chaining example #############################################

class IStart(object, metaclass=ABCMeta):
    pass

class IGeneric(object, metaclass=ABCMeta):
    pass

class ISpecific(IGeneric):
    pass

class IEnd(object, metaclass=ABCMeta):
    pass

class Start(object):
    pass

IStart.register(Start)

class IStartToISpecific(Adapter):
    pass

ISpecific.register(IStartToISpecific)

class IGenericToIEnd(Adapter):
    pass

IEnd.register(IGenericToIEnd)

#### EOF ######################################################################
